import os, atexit, itertools
try:
  import networkx as nx  # type: ignore
except ImportError:
  nx = None # graph won't work
from collections import defaultdict
from typing import Dict, List, Optional, Union
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, FusedOps, Op, OpType, LazyOp
from tinygrad.tensor import LazyBuffer
from tinygrad.helpers import GRAPH, GRAPHPATH, PRUNEGRAPH, DARKGRAPH, DEBUG, GlobalCounters
from tinygrad.runtime.lib import RawConst

# **** debugging and graphing ****

G = nx.DiGraph() if nx is not None else None
cnts: Dict[OpType, int] = defaultdict(int)
if DEBUG >= 2:
  def print_globalcounters():
    if GlobalCounters.time_sum_s == 0: return
    print(f"avg: {GlobalCounters.global_ops*1e-9/GlobalCounters.time_sum_s:8.2f} GFLOPS {GlobalCounters.global_mem*1e-9/GlobalCounters.time_sum_s:8.2f} GB/s",
          f"{' '*10}total: {GlobalCounters.kernel_count:5d} kernels {GlobalCounters.global_ops*1e-9:8.2f} GOPS {GlobalCounters.global_mem*1e-9:8.2f} GB {GlobalCounters.time_sum_s*1e3:8.2f} ms")
  atexit.register(print_globalcounters)
if GRAPH:
  def save_graph_exit():
    for k,v in cnts.items(): print(k, v)
    if PRUNEGRAPH: prune_graph()
    if DARKGRAPH:
      G.graph["graph"] = {"bgcolor" : "black"}
      G.graph["edge"] = {"color" : "white", "fontcolor" : "white"}
    print("saving", G)
    nx.drawing.nx_pydot.write_dot(G, f'{GRAPHPATH}.dot')

    with open(f'{GRAPHPATH}.dot', 'r') as f:
      content = f.read()
    index = content.rfind('}')
    if index != -1:
      new_content = content[:index] + generate_kernel_subgraphs_string() + content[index:]
      with open(f'{GRAPHPATH}.dot', 'w') as f:
        f.write(new_content)

    # -Gnslimit=100 can make it finish, but you won't like results
    os.system(f'dot -Tsvg {GRAPHPATH}.dot -o {GRAPHPATH}.svg')
    # os.system(f'dot -Tpng -Gdpi=300 {GRAPHPATH}.dot -o {GRAPHPATH}.png')
  atexit.register(save_graph_exit)

node_count = 0
def nm(x):
  global node_count
  if not hasattr(x, 'node_id'):
    setattr(x, 'node_id', node_count)
    node_count += 1
  return x.node_id

kernel_count = 0
def generate_kernel_subgraphs_string():
  for i in range(kernel_count):
    # find node with this value
    nodes = [node for (node, data) in G.nodes(data=True) if data.get('kernel') == i]
    assert len(nodes) == 1
    node = nodes[0]
    # flood fill all its ancestors
    for ancestor in nx.ancestors(G, node):
      if 'kernel' not in G.nodes[ancestor]:
        G.nodes[ancestor]['kernel'] = i

  # finally, create the string to be inserted into into the .dot file
  ret = "\n"
  for i in range(kernel_count):
    nodes = [node for (node, data) in G.nodes(data=True) if data.get('kernel') == i]
    ret += f"subgraph \"kernel {i}\" {{\n"
    ret += "cluster=true;\n" # https://graphviz.org/docs/attrs/cluster/
    ret += f"label=\"Kernel {i}\";\n"
    if DARKGRAPH:
      ret += "color=\"yellow\";\n"
      ret += "fontcolor=\"yellow\";\n"
    for node in nodes:
      ret += f"{node};\n"
    ret += "}\n\n"
  
  return ret

def get_sop(op: List[Op]):
  if len(op) <= 2: return '.'.join([str(y).split(".")[1] for y in op][::-1])
  if len(op) <= 4: return '.'.join([str(y).split(".")[1][0:3] for y in op][::-1])
  return str(len(op))

def str_dtype(dtyp):
  ret = str(dtyp)[7:]
  return "" if ret == 'float' else f"\n{ret}"

# GRAPH=3 has some other bug as well where intermediates are str8 duplicated, so stick w/ GRAPH=2 for now
# and when we do GRAPH=1, we actually start getting intermediate buffers instead of just intermediate ops
def log_intermediate_op(parent: Union[LazyBuffer, LazyOp], cur: Union[LazyBuffer, LazyOp], phantom=False):
  parent_lazyop = parent.op if type(parent) == LazyBuffer else parent
  cur_op = cur.op.op if type(cur) == LazyBuffer and not cur.realized else cur.op if type(cur) == LazyOp else None
  
  cur_in_src = parent_lazyop.src.count(cur) # for things like BiOps w/ the same source, like x * x
  G.add_edge(nm(cur), nm(parent), label=get_sop([parent_lazyop.op]) + (f", {cur_in_src}" if cur_in_src > 1 else ""))
  if phantom: G.edges[nm(cur), nm(parent)]['color'] = '#00FFFF9F' if DARKGRAPH else '#00000060'

  if type(cur) == LazyOp:
    child_shapes = [log_intermediate_op(cur, s) for s in cur.src]

    if type(cur_op) in [BinaryOps, UnaryOps, FusedOps]:
      inherited_shape = child_shapes[0]
    elif type(cur_op) == ReduceOps:
      inherited_shape = cur.arg
    elif type(cur_op) in [MovementOps, LoadOps, FusedOps]:
      raise NotImplementedError()
    else:
      raise NotImplementedError()

  else: pass # buffers are DAG leaves

  G.add_node(nm(cur))
  if any(attr not in G.nodes[nm(cur)] for attr in ['label', 'fillcolor', 'color', 'style', 'prunable']):
    dashed = type(cur) == LazyBuffer and not cur.realized and ((cur_op.op == LoadOps and hasattr(cur, "_backing")) or (hasattr(cur, "st") and not cur.st.contiguous))
    G.nodes[nm(cur)]['label'] = f"inter-buffer {cur.shape}{str_dtype(cur.dtype)}" if type(cur) == LazyBuffer else f"inter-op {inherited_shape}"
    G.nodes[nm(cur)]['fillcolor'] = (top_colors[type(cur_op)] + ('60' if phantom else ('80' if dashed else str()))) if type(cur_op) in top_colors else "#ffffff"
    G.nodes[nm(cur)]['color'] = 'white' if phantom ^ bool(DARKGRAPH) else 'black'
    G.nodes[nm(cur)]['style'] = ('filled, dashed' if dashed else 'filled')
    G.nodes[nm(cur)]['prunable'] = type(cur_op) in [LoadOps, MovementOps]

  return inherited_shape if type(cur) == LazyOp else cur.shape

top_colors = {LoadOps: '#FFFF80', UnaryOps: "#c0c0c0", ReduceOps: "#8080ff", BinaryOps: "#c0c0c0", MovementOps: "#80ff80", FusedOps: "#ff8080"}
def log_op(ret: LazyBuffer, ast: LazyOp, show_graph: Optional[bool] = None, phantom=False, kernel_op=False):
  if show_graph is None: show_graph = bool(GRAPH)
  if not DEBUG and not show_graph: return
  op: List[Op] = [x.op for x in ast.get_lazyops()]
  inp: List[LazyBuffer] = [x for x in ast.buffers if not isinstance(x.realized, RawConst)]
  # oporder = [LoadOps, FusedOps, ReduceOps, BinaryOps, UnaryOps, MovementOps]
  # optype = type(sorted(op, key=lambda x: oporder.index(type(x)))[0])
  optype = type(ast.op)
  cnts[optype] += 1
  if DEBUG >= 6: print(f"{op} : {', '.join([f'{x.shape}-<{nm(x)}>' for x in inp])} -> {ret.shape}-<{nm(ret)}>")

  if show_graph:
    for s in ast.src:
      log_intermediate_op(ret, s, phantom=phantom)
    if nm(ret) not in G.nodes: G.add_node(nm(ret))
    if kernel_op:
      global kernel_count
      G.nodes[nm(ret)]['kernel'] = kernel_count
      kernel_count += 1

    dashed = (optype == LoadOps and hasattr(ret, "_backing")) or (hasattr(ret, "st") and not ret.st.contiguous)  # type: ignore
    G.nodes[nm(ret)]['label'] = (str(set(x.shape for x in inp))+"\n"+str(ret.shape) if optype == ReduceOps else str(ret.shape))+str_dtype(ret.dtype)
    G.nodes[nm(ret)]['fillcolor'] = (top_colors[optype] + ('60' if phantom else ('80' if dashed else str()))) if optype in top_colors else "#ffffff"
    G.nodes[nm(ret)]['color'] = 'white' if phantom ^ bool(DARKGRAPH) else 'black'
    G.nodes[nm(ret)]['style'] = ('filled, dashed' if dashed else 'filled')
    G.nodes[nm(ret)]['prunable'] = optype in [LoadOps, MovementOps]

# prune movementops and loadops
def prune_graph():
  dead_nodes = []
  for n in G.nodes:
    if 'prunable' in G.nodes[n] and G.nodes[n]['prunable']:
      if G.in_degree(n) == 1:
        for (x,_),(_,y) in itertools.product(G.in_edges(n), G.out_edges(n)):
          G.add_edge(x, y)
          G.edges[x, y]['label'] = G.edges[n, y]['label']
        if 'kernel' in G.nodes[n]:
          for parent, _ in G.in_edges(n):
            G.nodes[parent]['kernel'] = G.nodes[n]['kernel']
        G.remove_edges_from(list(G.in_edges(n)) + list(G.out_edges(n)))
      else:
        G.add_edges_from([(x, y) for (x,_),(_,y) in itertools.product(G.in_edges(n), G.out_edges(n))])
      dead_nodes.append(n)
  G.remove_nodes_from(dead_nodes)
