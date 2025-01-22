#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::cmp::{min, max};

pub struct LCG {
    state: u64,
    // Parameters from "Numerical Recipes"
    // The modulus m is 2^64
    a: u64, // multiplier
    c: u64, // increment
}

impl LCG {
    pub fn new(seed: u64) -> Self {
        LCG {
            state: seed,
            a: 6364136223846793005,
            c: 1442695040888963407,
        }
    }

    // Avoid using the lower bits as they are less random
    pub fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(self.a).wrapping_add(self.c);
        self.state
    }

    // Get only the most significant 32 bits which are more random
    pub fn next_u32(&mut self) -> u32 {
        (self.next() >> 32) as u32
    }

    // Generate a random boolean
    pub fn rand_bool(&mut self) -> bool {
        (self.next() & (1 << 60)) != 0
    }

    // Choose a random index in [0, max[
    pub fn rand_idx(&mut self, max: usize) -> usize {
        (self.next_u32() as usize) % max
    }

    // Uniformly distributed usize in [min, max]
    pub fn rand_usize(&mut self, min: usize, max: usize) -> usize {
        assert!(max >= min);
        min + (self.next_u32() as usize) % (1 + max - min)
    }

    // Returns true with a specific percentage of probability
    pub fn pct_prob(&mut self, percent: usize) -> bool {
        let idx = self.rand_idx(100);
        idx < percent
    }

    // Pick a random element from a slice
    pub fn choice<'a, T>(&mut self, slice: &'a [T]) -> &'a T {
        assert!(!slice.is_empty());
        &slice[self.rand_idx(slice.len())]
    }
}

// Wait until we have interprocedural analysis working before introducing classes
/*
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Class {
    name: String,

    // List of fields
    fields: Vec<String>,

    // Types associated with each field
    field_types: Vec<Type>,

    // List of methods
    methods: HashMap<String, FunId>,

    // KISS, ignore for now
    // Constructor method
    //ctor: FunId,
}

impl std::fmt::Display for ClassId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Class@{}", self.0)
    }
}
*/

pub type FunId = usize;
pub type BlockId = usize;
pub type InsnId = usize;

// Type: Int, Nil, Class
#[derive(Debug, Copy, Clone, PartialEq)]
enum Type
{
    // Empty is the empty set (no info propagated yet or unreachable)
    Empty,
    Const(Value),
    Int,
    Bool,
    Any,
}

fn union(left: Type, right: Type) -> Type {
    match (left, right) {
        (Type::Any, _) | (_, Type::Any) => Type::Any,
        (Type::Empty, x) | (x, Type::Empty) => x,
        (l, r) if l == r => l,
        // Int
        (Type::Int, Type::Const(Value::Int(_))) | (Type::Const(Value::Int(_)), Type::Int) => Type::Int,
        (Type::Const(Value::Int(_)), Type::Const(Value::Int(_))) => Type::Int,
        (Type::Int, Type::Int) => Type::Int,
        // Bool
        (Type::Bool, Type::Const(Value::Bool(_))) | (Type::Const(Value::Bool(_)), Type::Bool) => Type::Bool,
        (Type::Const(Value::Bool(_)), Type::Const(Value::Bool(_))) => Type::Bool,
        (Type::Bool, Type::Bool) => Type::Bool,
        // Other
        _ => Type::Any,
    }
}

// Home of our interprocedural CFG
#[derive(Default, Debug)]
struct Program
{
    funs: Vec<Function>,

    blocks: Vec<Block>,

    insns: Vec<Insn>,

    // Main/entry function
    main: FunId,
}

impl Program {
    // Register a function and assign it an id
    pub fn new_fun(&mut self) -> (FunId, BlockId) {
        let mut fun = Function::default();
        let entry_block = self.new_block();
        fun.entry_block = entry_block;

        let id = self.funs.len();
        self.funs.push(fun);
        (id, entry_block)
    }

    // Register a block and assign it an id
    pub fn new_block(&mut self) -> BlockId {
        let id = self.blocks.len();
        self.blocks.push(Block::default());
        id
    }

    // Add an instruction to the program
    pub fn push_insn(&mut self, block: BlockId, op: Op) -> InsnId {
        let insn = Insn {
            op,
        };

        let id = self.insns.len();
        self.insns.push(insn);

        // Add the insn to the block
        self.blocks[block].insns.push(id);

        id
    }

    fn add_phi_arg(&mut self, insn_id: InsnId, block_id: BlockId, opnd: Opnd) {
        let insn = &mut self.insns[insn_id] ;
        match insn {
            Insn { op: Op::Phi { ins } } => ins.push((block_id, opnd)),
            _ => panic!("Can't append phi arg to non-phi {:?}", insn)
        }
    }
}

#[derive(Default, Debug)]
struct Function
{
    entry_block: BlockId,

    // We don't need an explicit list of blocks
}

#[derive(Default, Debug)]
struct Block
{
    // Do we need to keep the phi nodes separate?

    insns: Vec<InsnId>,
}

// Remove this if the only thing we store is op
#[derive(Debug)]
struct Insn
{
    op: Op,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug, Copy)]
pub enum Value {
    Nil,
    Int(i64),
    Bool(bool),
    Fun(FunId),
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum Opnd {
    // Constant
    Const(Value),

    // Output of a previous insn in a block
    // that dominates this one
    Insn(InsnId),
}

// Convenient constant operands
const NIL: Opnd = Opnd::Const(Value::Nil);
const ZERO: Opnd = Opnd::Const(Value::Int(0));
const ONE: Opnd = Opnd::Const(Value::Int(1));
const TWO: Opnd = Opnd::Const(Value::Int(2));
const FALSE: Opnd = Opnd::Const(Value::Bool(false));
const TRUE: Opnd = Opnd::Const(Value::Bool(true));

#[derive(Debug)]
enum Op
{
    Phi { ins: Vec<(BlockId, Opnd)> },
    Add { v0: Opnd, v1: Opnd },
    Mul { v0: Opnd, v1: Opnd },
    LessThan { v0: Opnd, v1: Opnd },
    IsNil { v: Opnd },

    // Start with a static send (no dynamic lookup)
    // to get the basics of the analysis working
    SendStatic { target: FunId, args: Vec<Opnd> },

    // TODO: wait until we have the interprocedural analysis working before tackling this
    // Send with a dynamic name lookup on `self`
    //Send { target: FunId, self: Opnd, args: Vec<Opnd> },

    // The caller blocks this function can return to are stored
    // on the Function object this instruction belongs to
    Return { val: Opnd, parent_fun: FunId },

    // Load a function parameter. Knows which function it belongs to so that we can more easily
    // flow type information in SCTP.
    Param { idx: usize, parent_fun: FunId },

    // Wait until we have basic interprocedural analysis working
    //GetIvar,
    //SetIvar,

    IfTrue { val: Opnd, then_block: BlockId, else_block: BlockId  },
    Jump { target: BlockId }
}

struct AnalysisResult {
    // Indexable by BlockId; indicates if the given block is potentially executable/reachable
    block_executable: Vec<bool>,

    // Indexable by InsnId; indicates the computed type of the result of the instruction
    // Instructions without outputs do not have types and their entries in this vector mean nothing
    insn_type: Vec<Type>,

    // Map of instructions to instructions that use them
    // uses[A] = { B, C } means that B and C both use A in their operands
    insn_uses: Vec<Vec<InsnId>>,

    // Number of iterations needed by the type analysis to
    // compute its result
    itr_count: usize,
}

// Sparse conditionall type propagation
#[inline(never)]
fn sctp(prog: &mut Program) -> AnalysisResult
{
    let graph = compute_uses(prog);
    let num_blocks = prog.blocks.len();
    let mut executable: Vec<bool> = Vec::with_capacity(num_blocks);
    executable.resize(num_blocks, false);

    let num_insns = prog.insns.len();
    let mut values: Vec<Type> = Vec::with_capacity(num_insns);
    values.resize(num_insns, Type::Empty);

    // Mark entry as executable
    let entry = prog.funs[prog.main].entry_block;
    executable[entry] = true;

    // Work list of instructions or blocks
    let mut block_worklist: VecDeque<BlockId> = VecDeque::new();
    let mut insn_worklist: VecDeque<InsnId> = VecDeque::from(prog.blocks[entry].insns.clone());

    let mut itr_count = 0;

    while block_worklist.len() > 0 || insn_worklist.len() > 0
    {
        while let Some(insn_id) = insn_worklist.pop_front() {
            itr_count += 1;

            let Insn {op} = &prog.insns[insn_id];
            // println!("looking at: {op:?}");
            // for (insn_id, insn) in prog.insns.iter().enumerate() {
            //     println!("{insn_id}: [{:?}] {:?}", values[insn_id], insn);
            // }
            // println!("----------------------------------");
            let old_value = values[insn_id];
            let value_of = |opnd: &Opnd| -> Type {
                match opnd {
                    Opnd::Const(v) => Type::Const(*v),
                    Opnd::Insn(insn_id) => values[*insn_id],
                }
            };
            // Handle control instructions first; they do not have a value
            if let Op::IfTrue { val, then_block, else_block } = op {
                match value_of(val) {
                    Type::Const(Value::Bool(false)) => block_worklist.push_back(*else_block),
                    Type::Const(Value::Bool(true)) => block_worklist.push_back(*then_block),
                    _ => {
                        block_worklist.push_back(*then_block);
                        block_worklist.push_back(*else_block);
                    }
                }
                continue;
            };
            if let Op::Jump { target } = op {
                block_worklist.push_back(*target);
                continue;
            };
            if let Op::Return { val, parent_fun } = op {
                // TODO(max): Should we instead be extending conditionally?
                insn_worklist.extend(&graph.insn_uses[insn_id]);
                continue;
            };
            // Now handle expression-like instructions
            let new_value = match op {
                Op::Add {v0, v1} => {
                    match (value_of(v0), value_of(v1)) {
                        (Type::Empty, _) | (_, Type::Empty) => Type::Empty,
                        (Type::Const(Value::Int(l)), Type::Const(Value::Int(r))) => Type::Const(Value::Int(l+r)),
                        (l, r) if union(l, r) == Type::Int => Type::Int,
                        _ => Type::Any,
                    }
                }
                Op::Mul {v0, v1} => {
                    match (value_of(v0), value_of(v1)) {
                        (Type::Empty, _) | (_, Type::Empty) => Type::Empty,
                        (Type::Const(Value::Int(l)), Type::Const(Value::Int(r))) => Type::Const(Value::Int(l*r)),
                        (l, r) if union(l, r) == Type::Int => Type::Int,
                        _ => Type::Any,
                    }
                }
                Op::LessThan {v0, v1} => {
                    match (value_of(v0), value_of(v1)) {
                        (Type::Empty, _) | (_, Type::Empty) => Type::Empty,
                        (Type::Const(Value::Int(l)), Type::Const(Value::Int(r))) => Type::Const(Value::Bool(l<r)),
                        (l, r) if union(l, r) == Type::Int => Type::Bool,
                        _ => Type::Any,
                    }
                }
                Op::IsNil { v } => {
                    match value_of(v) {
                    Type::Empty => Type::Empty,
                        Type::Const(Value::Nil) => Type::Const(Value::Bool(true)),
                        Type::Const(_) | Type::Int | Type::Bool => Type::Const(Value::Bool(false)),
                        _ => Type::Bool,
                    }
                }
                Op::Phi { ins } => {
                    // Only take into account operands coming from from reachable blocks
                    ins.iter().fold(Type::Empty, |acc, (block_id, opnd)| if executable[*block_id] { union(acc, value_of(opnd)) } else { acc })
                }
                Op::Param { idx, parent_fun } => {
                    graph.flows_to[insn_id].iter().fold(Type::Empty, |acc, opnd| union(acc, value_of(opnd)))
                }
                Op::SendStatic { target, args } => {
                    block_worklist.push_back(prog.funs[*target].entry_block);
                    graph.flows_to[insn_id].iter().fold(Type::Empty, |acc, opnd| union(acc, value_of(opnd)))
                }
                _ => todo!("op not yet supported {:?}", op),
            };
            if union(old_value, new_value) != old_value {
                values[insn_id] = new_value;
                insn_worklist.extend(&graph.insn_uses[insn_id]);
            }
        }

        while let Some(block_id) = block_worklist.pop_front() {
            itr_count += 1;

            if !executable[block_id] {
                executable[block_id] = true;
                insn_worklist.extend(&prog.blocks[block_id].insns);
            }
        }
    }

    AnalysisResult {
        block_executable: executable,
        insn_type: values,
        insn_uses: graph.insn_uses,
        itr_count
    }
}

struct CallGraph {
    // Map of InsnId -> instructions that use that insn
    insn_uses: Vec<Vec<InsnId>>,
    // Map of InsnId -> operands that flow to that insn
    // Flow goes from send arguments to function parameters and from return values to send results
    flows_to: Vec<Vec<Opnd>>,
}

#[inline(never)]
fn compute_uses(prog: &mut Program) -> CallGraph {
    // Map of functions to instructions that called them
    let num_funs = prog.funs.len();
    let mut called_by: Vec<HashSet<InsnId>> = Vec::with_capacity(num_funs);
    called_by.resize(num_funs, HashSet::new());
    for (insn_id, insn) in prog.insns.iter().enumerate() {
        let Insn { op, .. } = insn;
        match op {
            Op::SendStatic { target, args } => {
                called_by[*target].insert(insn_id);
            }
            _ => {}
        }
    }
    // Map of instructions to instructions that use them
    // uses[A] = { B, C } means that B and C both use A in their operands
    let num_insns = prog.insns.len();
    let mut uses: Vec<HashSet<InsnId>> = Vec::with_capacity(num_insns);
    uses.resize(num_insns, HashSet::new());
    // Map of InsnId -> instructions that flow to that insn
    let mut flows_to: Vec<HashSet<Opnd>> = Vec::with_capacity(num_insns);
    flows_to.resize(num_insns, HashSet::new());
    for (insn_id, insn) in prog.insns.iter().enumerate() {
        let mut mark_use = |user: InsnId, opnd: &Opnd| {
            match opnd {
                Opnd::Insn(used) => {
                    uses[*used].insert(user);
                }
                _ => {}
            }
        };
        match &insn.op {
            Op::Phi { ins } => {
                for (_, opnd) in ins {
                    mark_use(insn_id, opnd);
                }
            }
            Op::Add { v0, v1 } | Op::Mul {v0, v1 } | Op::LessThan { v0, v1 } => {
                mark_use(insn_id, v0);
                mark_use(insn_id, v1);
            }
            Op::IsNil { v } => {
                mark_use(insn_id, v);
            }
            Op::SendStatic { args, .. } => {
                for opnd in args {
                    mark_use(insn_id, opnd);
                }
            }
            Op::Return { val, parent_fun } => {
                mark_use(insn_id, val);
                for caller in &called_by[*parent_fun] {
                    mark_use(*caller, &Opnd::Insn(insn_id));
                    flows_to[*caller].insert(val.clone());
                }
            }
            Op::Param { idx, parent_fun } => {
                for caller in &called_by[*parent_fun] {
                    mark_use(insn_id, &Opnd::Insn(*caller));
                    match &prog.insns[*caller].op {
                        Op::SendStatic { args, .. } => {
                            if *idx < args.len() {
                                flows_to[insn_id].insert(args[*idx].clone());
                            }
                        }
                        op => panic!("Only send should call function; found {op:?}"),
                    }
                }
            }
            Op::IfTrue { val, .. } => {
                mark_use(insn_id, val);
            }
            Op::Jump { .. } => {}
        }
    }
    CallGraph {
        insn_uses: uses.into_iter().map(|set| set.into_iter().collect()).collect(),
        flows_to: flows_to.into_iter().map(|set| set.into_iter().collect()).collect(),
    }
}

// Generate a random acyclic call graph
fn random_dag(rng: &mut LCG, num_nodes: usize, min_parents: usize, max_parents: usize) -> Vec<Vec<usize>>
{
    let mut callees: Vec<Vec<FunId>> = Vec::new();

    callees.resize(num_nodes, Vec::default());

    // For each node except the root
    // Node 0 is the root node and has no incoming edge
    for node_idx in 1..num_nodes
    {
        // Choose random number of parents
        let num_parents = rng.rand_usize(min_parents, min(max_parents, node_idx));

        // For each parent
        for i in 0..num_parents {
            // Select a random previous node as the parent
            let p_idx = rng.rand_usize(0, node_idx - 1);
            callees[p_idx].push(node_idx);
        }
    }

    callees
}

#[inline(never)]
fn gen_torture_test(num_funs: usize) -> Program
{
    let mut rng = LCG::new(1337);

    let callees = random_dag(&mut rng, num_funs, 1, 10);
    assert!(callees[0].len() > 0);

    let mut prog = Program::default();

    // The first function is the root node of the graph
    prog.main = 0;

    // For each function to be generated
    for fun_id in 0..num_funs {
        let (f_id, entry_block) = prog.new_fun();
        assert!(f_id == fun_id);

         // List of callees for this function
        let callees = &callees[fun_id];

        // If this is a leaf method
        if callees.is_empty() {
            // Return a constant
            let const_val = if rng.rand_bool() {
                Value::Nil
            } else {
                Value::Int(rng.rand_usize(0, 500) as i64)
            };

            prog.push_insn(
                entry_block,
                Op::Return { val: Opnd::Const(const_val), parent_fun: fun_id }
            );
        } else {
            let mut last_block = entry_block;
            let mut sum_val = ZERO;

            // Call the callees and do nothing with the return value for now
            for callee_id in callees {
                let nil_block = prog.new_block();
                let int_block = prog.new_block();
                let sum_block = prog.new_block();

                let call_insn = prog.push_insn(
                    last_block,
                    Op::SendStatic { target: *callee_id, args: vec![] }
                );
                let isnil_insn = prog.push_insn(last_block, Op::IsNil { v: Opnd::Insn(call_insn) });
                prog.push_insn(last_block, Op::IfTrue { val: Opnd::Insn(isnil_insn), then_block: nil_block, else_block: int_block });

                // Both branches go to the sum block
                prog.push_insn(nil_block, Op::Jump { target: sum_block });
                prog.push_insn(int_block, Op::Jump { target: sum_block });

                // Compute the sum
                let phi_id = prog.push_insn(sum_block, Op::Phi { ins: vec![(nil_block, ZERO), (int_block, Opnd::Insn(call_insn))] });
                prog.push_insn(sum_block, Op::Add { v0: sum_val.clone(), v1: Opnd::Insn(phi_id) });
                sum_val = Opnd::Insn(phi_id);
            }

            prog.push_insn(
                last_block,
                Op::Return { val: sum_val, parent_fun: fun_id }
            );
        }
    }

    prog
}

fn main()
{
    // TODO:
    // 1. Start with intraprocedural analysis
    // 2. Get interprocedural analysis working with direct send (no classes, no native methods)
    // 3. Once confident that interprocedural analysis is working, then add classes and objects





    let mut prog = gen_torture_test(10_000);

    use std::time::Instant;
    let start_time = Instant::now();
    let result = sctp(&mut prog);
    let duration = start_time.elapsed();
    let time_ms = duration.as_secs_f64() * 1000.0;

    // Check that all functions marked executable
    for fun in &prog.funs {
        let entry_id = fun.entry_block;
        if !result.block_executable[entry_id] {
            panic!("all function entry blocks should be executable");
        }
    }

    // Check that the main return type is integer
    for (insn_id, insn) in prog.insns.iter().enumerate() {
        if let Op::Return { val: Opnd::Insn(ret_id), parent_fun } = &insn.op {
            if *parent_fun == prog.main {
                let ret_type = result.insn_type[*ret_id];
                println!("main return type: {:?}", ret_type);

                if ret_type != Type::Int {
                    panic!("output type should be integer");
                }
            }
        }
    }

    println!("analysis time: {:.1} ms", time_ms);
    println!("itr count: {}", result.itr_count);
}

#[cfg(test)]
mod union_tests {
    use super::*;

    #[test]
    fn test_any() {
        assert_eq!(union(Type::Any, Type::Any), Type::Any);
        assert_eq!(union(Type::Any, Type::Int), Type::Any);
        assert_eq!(union(Type::Any, Type::Empty), Type::Any);
        assert_eq!(union(Type::Any, Type::Int), Type::Any);
        assert_eq!(union(Type::Any, Type::Bool), Type::Any);
        assert_eq!(union(Type::Any, Type::Const(Value::Int(5))), Type::Any);
        assert_eq!(union(Type::Any, Type::Const(Value::Bool(true))), Type::Any);
    }

    #[test]
    fn test_empty() {
        assert_eq!(union(Type::Empty, Type::Any), Type::Any);
        assert_eq!(union(Type::Empty, Type::Int), Type::Int);
        assert_eq!(union(Type::Empty, Type::Empty), Type::Empty);
        assert_eq!(union(Type::Empty, Type::Const(Value::Int(5))), Type::Const(Value::Int(5)));
        assert_eq!(union(Type::Empty, Type::Int), Type::Int);
        assert_eq!(union(Type::Empty, Type::Bool), Type::Bool);
        assert_eq!(union(Type::Empty, Type::Const(Value::Bool(true))), Type::Const(Value::Bool(true)));
    }

    #[test]
    fn test_const() {
        assert_eq!(union(Type::Const(Value::Int(3)), Type::Const(Value::Int(3))), Type::Const(Value::Int(3)));
        assert_eq!(union(Type::Const(Value::Bool(true)), Type::Const(Value::Bool(true))), Type::Const(Value::Bool(true)));
        assert_eq!(union(Type::Const(Value::Int(3)), Type::Const(Value::Bool(true))), Type::Any);
    }

    #[test]
    fn test_type() {
        assert_eq!(union(Type::Const(Value::Int(3)), Type::Const(Value::Int(4))), Type::Int);
        assert_eq!(union(Type::Const(Value::Int(3)), Type::Int), Type::Int);

        assert_eq!(union(Type::Const(Value::Bool(true)), Type::Const(Value::Bool(false))), Type::Bool);
        assert_eq!(union(Type::Const(Value::Bool(true)), Type::Bool), Type::Bool);

        assert_eq!(union(Type::Int, Type::Bool), Type::Any);
    }
}

#[cfg(test)]
mod compute_uses_tests {
    use super::*;

    fn prog_with_empty_fun() -> (Program, FunId, BlockId) {
        let mut prog = Program::default();
        prog.main = 0;
        let (fun_id, block_id) = prog.new_fun();
        (prog, fun_id, block_id)
    }

    #[test]
    fn test_return() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let ret_id = prog.push_insn(block_id, Op::Return { val: Opnd::Insn(add_id), parent_fun: fun_id });
        let CallGraph { insn_uses, .. } = compute_uses(&mut prog);
        assert_eq!(insn_uses[add_id], vec![ret_id]);
        assert_eq!(insn_uses[ret_id], vec![]);
    }

    #[test]
    fn test_add() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add0_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let add1_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Insn(add0_id), v1: Opnd::Insn(add0_id) });
        let CallGraph { insn_uses, .. } = compute_uses(&mut prog);
        assert_eq!(insn_uses[add0_id], vec![add1_id]);
        assert_eq!(insn_uses[add1_id], vec![]);
    }

    #[test]
    fn test_phi() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let phi_id = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::Insn(add_id))] });
        let CallGraph { insn_uses, .. } = compute_uses(&mut prog);
        assert_eq!(insn_uses[add_id], vec![phi_id]);
        assert_eq!(insn_uses[phi_id], vec![]);
    }

    #[test]
    fn test_send() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let (target, target_entry) = prog.new_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let send_id = prog.push_insn(block_id, Op::SendStatic { target, args: vec![Opnd::Insn(add_id)] });
        let CallGraph { insn_uses, .. } = compute_uses(&mut prog);
        assert_eq!(insn_uses[add_id], vec![send_id]);
        assert_eq!(insn_uses[send_id], vec![]);
    }

    #[test]
    fn test_send_uses_return() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let (target, target_entry) = prog.new_fun();
        let ret_id = prog.push_insn(target_entry, Op::Return { val: Opnd::Const(Value::Int(5)), parent_fun: target });
        let send_id = prog.push_insn(block_id, Op::SendStatic { target, args: vec![Opnd::Const(Value::Int(4))] });
        let CallGraph { insn_uses, .. } = compute_uses(&mut prog);
        assert_eq!(insn_uses[send_id], vec![]);
        assert_eq!(insn_uses[ret_id], vec![send_id]);
    }

    #[test]
    fn test_param_uses_send() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let (target, target_entry) = prog.new_fun();
        let param_id = prog.push_insn(target_entry, Op::Param { idx: 0, parent_fun: target });
        let ret_id = prog.push_insn(target_entry, Op::Return { val: Opnd::Const(Value::Int(5)), parent_fun: target });
        let send_id = prog.push_insn(block_id, Op::SendStatic { target, args: vec![Opnd::Const(Value::Int(4))] });
        let CallGraph { insn_uses, .. } = compute_uses(&mut prog);
        assert_eq!(insn_uses[send_id], vec![param_id]);
    }

    #[test]
    fn test_iftrue() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let iftrue_id = prog.push_insn(block_id, Op::IfTrue { val: Opnd::Insn(add_id), then_block: 3, else_block: 4 });
        let CallGraph { insn_uses, .. } = compute_uses(&mut prog);
        assert_eq!(insn_uses[add_id], vec![iftrue_id]);
        assert_eq!(insn_uses[iftrue_id], vec![]);
    }
}

#[cfg(test)]
mod sctp_tests {
    use super::*;

    fn prog_with_empty_fun() -> (Program, FunId, BlockId) {
        let mut prog = Program::default();
        prog.main = 0;
        let (fun_id, block_id) = prog.new_fun();
        (prog, fun_id, block_id)
    }

    #[test]
    fn test_add_const() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let result = sctp(&mut prog);
        assert_eq!(result.insn_type[add_id], Type::Const(Value::Int(7)));
    }

    #[test]
    fn test_add_insn() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add0_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let add1_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Insn(add0_id), v1: Opnd::Const(Value::Int(5)) });
        let result = sctp(&mut prog);
        assert_eq!(result.insn_type[add1_id], Type::Const(Value::Int(12)));
    }

    #[test]
    fn test_isnil_non_nil_const() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let isnil_id = prog.push_insn(block_id, Op::IsNil { v: Opnd::Const(Value::Int(3)) });
        let result = sctp(&mut prog);
        assert_eq!(result.insn_type[isnil_id], Type::Const(Value::Bool(false)));
    }

    #[test]
    fn test_isnil_nil() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let isnil_id = prog.push_insn(block_id, Op::IsNil { v: NIL });
        let result = sctp(&mut prog);
        assert_eq!(result.insn_type[isnil_id], Type::Const(Value::Bool(true)));
    }

    #[test]
    fn test_isnil_int_type() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let phi_id = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::Const(Value::Int(3))), (block_id, Opnd::Const(Value::Int(4)))] });
        let isnil_id = prog.push_insn(block_id, Op::IsNil { v: Opnd::Insn(phi_id) });
        let result = sctp(&mut prog);
        assert_eq!(result.insn_type[phi_id], Type::Int);
        assert_eq!(result.insn_type[isnil_id], Type::Const(Value::Bool(false)));
    }

    #[test]
    fn test_isnil_any() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let phi_id = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::Const(Value::Bool(true))), (block_id, Opnd::Const(Value::Int(4)))] });
        let isnil_id = prog.push_insn(block_id, Op::IsNil { v: Opnd::Insn(phi_id) });
        let result = sctp(&mut prog);
        assert_eq!(result.insn_type[phi_id], Type::Any);
        assert_eq!(result.insn_type[isnil_id], Type::Bool);
    }

    #[test]
    fn test_less_than() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let lt0_id = prog.push_insn(block_id, Op::LessThan { v0: Opnd::Const(Value::Int(7)), v1: Opnd::Const(Value::Int(8)) });
        let lt1_id = prog.push_insn(block_id, Op::LessThan { v0: Opnd::Const(Value::Int(8)), v1: Opnd::Const(Value::Int(8)) });
        let result = sctp(&mut prog);
        assert_eq!(result.insn_type[lt0_id], Type::Const(Value::Bool(true)));
        assert_eq!(result.insn_type[lt1_id], Type::Const(Value::Bool(false)));
    }

    #[test]
    fn test_phi_same_const() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let phi_id = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::Insn(add_id)), (block_id, Opnd::Const(Value::Int(7)))] });
        let result = sctp(&mut prog);
        assert_eq!(result.insn_type[phi_id], Type::Const(Value::Int(7)));
    }

    #[test]
    fn test_phi_different_const() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let phi_id = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::Insn(add_id)), (block_id, Opnd::Const(Value::Int(8)))] });
        let result = sctp(&mut prog);
        assert_eq!(result.insn_type[phi_id], Type::Int);
    }

    #[test]
    fn test_phi_different_type() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let phi_id = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::Insn(add_id)), (block_id, TRUE)] });
        let result = sctp(&mut prog);
        assert_eq!(result.insn_type[phi_id], Type::Any);
    }

    #[test]
    fn test_iftrue_any() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let phi_id = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, ONE), (block_id, TWO)] });
        let then_block = prog.new_block();
        let else_block = prog.new_block();
        let iftrue_id = prog.push_insn(block_id, Op::IfTrue { val: Opnd::Insn(phi_id), then_block, else_block });
        let result = sctp(&mut prog);
        assert_eq!(result.insn_type[phi_id], Type::Int);
        assert_eq!(result.block_executable[then_block], true);
        assert_eq!(result.block_executable[else_block], true);
    }

    #[test]
    fn test_iftrue_const_then() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let then_block = prog.new_block();
        let else_block = prog.new_block();
        let iftrue_id = prog.push_insn(block_id, Op::IfTrue { val: TRUE, then_block, else_block });
        let result = sctp(&mut prog);
        assert_eq!(result.block_executable[then_block], true);
        assert_eq!(result.block_executable[else_block], false);
    }

    #[test]
    fn test_iftrue_const_else() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let then_block = prog.new_block();
        let else_block = prog.new_block();
        let iftrue_id = prog.push_insn(block_id, Op::IfTrue { val: FALSE, then_block, else_block });
        let result = sctp(&mut prog);
        assert_eq!(result.block_executable[then_block], false);
        assert_eq!(result.block_executable[else_block], true);
    }

    #[test]
    fn test_jump() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let target = prog.new_block();
        let iftrue_id = prog.push_insn(block_id, Op::Jump { target });
        let result = sctp(&mut prog);
        assert_eq!(result.block_executable[target], true);
    }

    #[test]
    fn test_one_return_flows_to_send() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let (target, target_entry) = prog.new_fun();
        let return_id = prog.push_insn(target_entry, Op::Return { val: Opnd::Const(Value::Int(5)), parent_fun: target });
        let send_id = prog.push_insn(block_id, Op::SendStatic { target, args: vec![] });
        let result = sctp(&mut prog);
        assert_eq!(result.block_executable[target_entry], true);
        assert_eq!(result.insn_type[send_id], Type::Const(Value::Int(5)));
    }

    #[test]
    fn test_one_send_flows_to_param() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let (target, target_entry) = prog.new_fun();
        let param_id = prog.push_insn(target_entry, Op::Param { idx: 0, parent_fun: target });
        let return_id = prog.push_insn(target_entry, Op::Return { val: Opnd::Insn(param_id), parent_fun: target });
        let send_id = prog.push_insn(block_id, Op::SendStatic { target, args: vec![Opnd::Const(Value::Int(5))] });
        let result = sctp(&mut prog);
        assert_eq!(result.block_executable[target_entry], true);
        assert_eq!(result.insn_type[param_id], Type::Const(Value::Int(5)));
    }

    #[test]
    fn test_two_sends_flow_to_param() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let (target, target_entry) = prog.new_fun();
        let param_id = prog.push_insn(target_entry, Op::Param { idx: 0, parent_fun: target });
        let return_id = prog.push_insn(target_entry, Op::Return { val: Opnd::Insn(param_id), parent_fun: target });
        let send0_id = prog.push_insn(block_id, Op::SendStatic { target, args: vec![Opnd::Const(Value::Int(5))] });
        let send1_id = prog.push_insn(block_id, Op::SendStatic { target, args: vec![Opnd::Const(Value::Int(6))] });
        let result = sctp(&mut prog);
        assert_eq!(result.block_executable[target_entry], true);
        assert_eq!(result.insn_type[param_id], Type::Int);
    }

    #[test]
    fn test_send_multiple_args() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let (target, target_entry) = prog.new_fun();
        let param0_id = prog.push_insn(target_entry, Op::Param { idx: 0, parent_fun: target });
        let param1_id = prog.push_insn(target_entry, Op::Param { idx: 1, parent_fun: target });
        let add_id = prog.push_insn(target_entry, Op::Add { v0: Opnd::Insn(param0_id), v1: Opnd::Insn(param1_id) });
        let return_id = prog.push_insn(target_entry, Op::Return { val: Opnd::Insn(add_id), parent_fun: target });
        let send_id = prog.push_insn(block_id, Op::SendStatic { target, args: vec![Opnd::Const(Value::Int(3)), Opnd::Const(Value::Int(4))] });
        let result = sctp(&mut prog);
        assert_eq!(result.block_executable[target_entry], true);
        assert_eq!(result.insn_type[param0_id], Type::Const(Value::Int(3)));
        assert_eq!(result.insn_type[param1_id], Type::Const(Value::Int(4)));
        assert_eq!(result.insn_type[send_id], Type::Const(Value::Int(7)));
    }

    #[test]
    fn test_loop_sum() {
        /*
        entry:
            Jump body
        body:
            n = Phi (entry: 0, body: n_inc)
            n_inc = Add(n, 1)
            cond = LessThan n, 100
            IfTrue cond, body, end
        end:
            Return n
        */
        let (mut prog, fun_id, entry_id) = prog_with_empty_fun();
        let body_id = prog.new_block();
        let end_id = prog.new_block();
        prog.push_insn(entry_id, Op::Jump { target: body_id });
        let n = prog.push_insn(body_id, Op::Phi { ins: vec![(entry_id, ZERO)] });
        let n_inc = prog.push_insn(body_id, Op::Add { v0: Opnd::Insn(n), v1: ONE });
        prog.add_phi_arg(n, body_id, Opnd::Insn(n_inc));
        let cond = prog.push_insn(body_id, Op::LessThan { v0: Opnd::Insn(n), v1: Opnd::Const(Value::Int(100)) });
        prog.push_insn(body_id, Op::IfTrue { val: Opnd::Insn(cond), then_block: body_id, else_block: end_id });
        prog.push_insn(end_id, Op::Return { val: Opnd::Insn(n), parent_fun: fun_id });
        let result = sctp(&mut prog);
        assert_eq!(result.block_executable[entry_id], true);
        assert_eq!(result.block_executable[body_id], true);
        assert_eq!(result.block_executable[end_id], true);
        assert_eq!(result.insn_type[n], Type::Int);
        assert_eq!(result.insn_type[n_inc], Type::Int);
        assert_eq!(result.insn_type[cond], Type::Bool);
    }

    #[test]
    fn test_factorial() {
        // fact(n)
        //   if n < 2
        //     return 1
        //   else
        //     n * fact(n-1)

        /*
        entry:
            n = Param(0, fact)
            lt = LessThan n, Const(2)
            IfTrue lt, early_exit, do_mul
        early_exit:
            Return Const(1)
        do_mul:
            sub = Add n, Const(-1)
            rec = SendStatic fact, [ sub ]
            mul = Mul n, rec
            Return mul

        ...

        SendStatic fact, [ Const(5) ]
        */
        let (mut prog, fun_id, entry_id) = prog_with_empty_fun();
        let (fact_id, fact_entry) = prog.new_fun();
        let outside_call = prog.push_insn(entry_id, Op::SendStatic { target: fact_id, args: vec![Opnd::Const(Value::Int(5))] });
        let n = prog.push_insn(fact_entry, Op::Param { idx: 0, parent_fun: fact_id });
        let lt = prog.push_insn(fact_entry, Op::LessThan { v0: Opnd::Insn(n), v1: TWO });
        let early_exit_id = prog.new_block();
        let do_mul_id = prog.new_block();
        prog.push_insn(fact_entry, Op::IfTrue { val: Opnd::Insn(lt), then_block: early_exit_id, else_block: do_mul_id });
        prog.push_insn(early_exit_id, Op::Return { val: ONE, parent_fun: fact_id });
        let sub = prog.push_insn(do_mul_id, Op::Add { v0: Opnd::Insn(n), v1: Opnd::Const(Value::Int(-1)) });
        let rec = prog.push_insn(do_mul_id, Op::SendStatic { target: fact_id, args: vec![Opnd::Insn(sub)] });
        let mul = prog.push_insn(do_mul_id, Op::Mul { v0: Opnd::Insn(n), v1: Opnd::Insn(rec) });
        prog.push_insn(do_mul_id, Op::Return { val: Opnd::Insn(mul), parent_fun: fact_id });
        let result = sctp(&mut prog);
        assert_eq!(result.block_executable[entry_id], true);
        assert_eq!(result.block_executable[fact_entry], true);
        assert_eq!(result.block_executable[early_exit_id], true);
        assert_eq!(result.block_executable[do_mul_id], true);
        assert_eq!(result.insn_type[outside_call], Type::Int);
        assert_eq!(result.insn_type[n], Type::Int);
        assert_eq!(result.insn_type[lt], Type::Bool);
        assert_eq!(result.insn_type[sub], Type::Int);
        assert_eq!(result.insn_type[rec], Type::Int);
        assert_eq!(result.insn_type[mul], Type::Int);
    }

    #[test]
    fn test_fib() {
        /*
        fib(n)
          if n < 2
            return n
          else
            return fib(n-1) + fib(n-2)
        */

        /*
        entry:
            n = Param(0, fib)
            lt = LessThan n, 2
            IfTrue lt, early_exit, do_add
        early_exit:
            Return n
        do_add:
            sub1 = Add n, Const(-1)
            rec1 = SendStatic fib, [ sub1 ]
            sub2 = Add n, Const(-2)
            rec2 = SendStatic fib, [ sub2 ]
            add = Add rec1, rec2
            Return add

        ...

        SendStatic fib, [ Const(100) ]
        */
        let (mut prog, _, entry_id) = prog_with_empty_fun();
        let (fib_id, fib_entry) = prog.new_fun();
        let outside_call = prog.push_insn(entry_id, Op::SendStatic { target: fib_id, args: vec![Opnd::Const(Value::Int(100))] });
        let n = prog.push_insn(fib_entry, Op::Param { idx: 0, parent_fun: fib_id });
        let lt = prog.push_insn(fib_entry, Op::LessThan { v0: Opnd::Insn(n), v1: TWO });
        let early_exit_id = prog.new_block();
        let do_add_id = prog.new_block();
        prog.push_insn(fib_entry, Op::IfTrue { val: Opnd::Insn(lt), then_block: early_exit_id, else_block: do_add_id });
        prog.push_insn(early_exit_id, Op::Return { val: Opnd::Insn(n), parent_fun: fib_id });
        let sub1 = prog.push_insn(do_add_id, Op::Add { v0: Opnd::Insn(n), v1: Opnd::Const(Value::Int(-1)) });
        let rec1 = prog.push_insn(do_add_id, Op::SendStatic { target: fib_id, args: vec![Opnd::Insn(sub1)] });
        let sub2 = prog.push_insn(do_add_id, Op::Add { v0: Opnd::Insn(n), v1: Opnd::Const(Value::Int(-2)) });
        let rec2 = prog.push_insn(do_add_id, Op::SendStatic { target: fib_id, args: vec![Opnd::Insn(sub2)] });
        let add = prog.push_insn(do_add_id, Op::Add { v0: Opnd::Insn(rec1), v1: Opnd::Insn(rec2) });
        prog.push_insn(do_add_id, Op::Return { val: Opnd::Insn(add), parent_fun: fib_id });
        let result = sctp(&mut prog);
        assert_eq!(result.block_executable[entry_id], true);
        assert_eq!(result.block_executable[fib_entry], true);
        assert_eq!(result.block_executable[early_exit_id], true);
        assert_eq!(result.block_executable[do_add_id], true);
        assert_eq!(result.insn_type[outside_call], Type::Int);
        assert_eq!(result.insn_type[n], Type::Int);
        assert_eq!(result.insn_type[lt], Type::Bool);
        assert_eq!(result.insn_type[sub1], Type::Int);
        assert_eq!(result.insn_type[rec1], Type::Int);
        assert_eq!(result.insn_type[sub2], Type::Int);
        assert_eq!(result.insn_type[rec2], Type::Int);
        assert_eq!(result.insn_type[add], Type::Int);
    }
}
