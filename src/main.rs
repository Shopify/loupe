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
#[derive(Default, Debug, Copy, Clone, PartialEq)]
enum Type
{
    // Bottom is the empty set (no info propagated yet)
    #[default]
    Bottom,
    Const(Value),
    Top,
}

fn meet(left: Type, right: Type) -> Type {
    match (left, right) {
        (Type::Top, x) | (x, Type::Top) => x,
        (l, r) if l == r => l,
        (Type::Bottom, x) | (x, Type::Bottom) => Type::Bottom,
        (Type::Const(l), Type::Const(r)) => Type::Top,
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
}

#[derive(Default, Debug)]
struct Function
{
    entry_block: BlockId,

    // These are the blocks this function can return to
    // If we add a block to this, we need to update propagation from return block(s)
    cont_blocks: Vec<BlockId>,

    // We don't need an explicit list of blocks
}

#[derive(Default, Debug)]
struct Block
{
    // Do we need to keep the phi nodes separate?

    insns: Vec<InsnId>,
}

#[derive(Debug)]
struct Insn
{
    op: Op,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug, Copy)]
pub enum Value {
    Nil,
    Int(i64),
    Fun(FunId),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Opnd {
    // Constant
    Const(Value),

    // Output of a previous insn in a block
    // that dominates this one
    InsnOut(InsnId),
}

#[derive(Debug)]
enum Op
{
    Phi { ins: Vec<(BlockId, Opnd)> },
    Add { v0: Opnd, v1: Opnd },

    // Start with a static send (no dynamic lookup)
    // to get the basics of the analysis working
    // ret_block is the block that we return to after the call
    SendStatic { target: FunId, args: Vec<Opnd>, ret_block: BlockId },

    // TODO: wait until we have the interprocedural analysis working before tackling this
    // Send with a dynamic name lookup on `self`
    //Send { target: FunId, self: Opnd, args: Vec<Opnd>, ret_block: BlockId },

    // The caller blocks this function can return to are stored
    // on the Function object this instruction belongs to
    Return { val: Opnd, parent_fun: FunId },

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
}

// Sparse conditionall type propagation
fn sctp(prog: &mut Program) -> AnalysisResult
{
    let uses = compute_uses(prog);
    let num_blocks = prog.blocks.len();
    let mut executable: Vec<bool> = Vec::with_capacity(num_blocks);
    executable.resize(num_blocks, false);

    let num_insns = prog.insns.len();
    let mut values: Vec<Type> = Vec::with_capacity(num_insns);
    values.resize(num_insns, Type::Top);

    // Mark entry as executable
    let entry = prog.funs[prog.main].entry_block;
    executable[entry] = true;

    // Work list of instructions or blocks
    let mut block_worklist: VecDeque<BlockId> = VecDeque::new();
    let mut insn_worklist: VecDeque<InsnId> = VecDeque::from(prog.blocks[entry].insns.clone());

    while block_worklist.len() > 0 || insn_worklist.len() > 0
    {
        while let Some(insn_id) = insn_worklist.pop_front() {
            let Insn {op} = &prog.insns[insn_id];
            let old_value = values[insn_id];
            let value_of = |opnd: Opnd| -> Type {
                match opnd {
                    Opnd::Const(v) => Type::Const(v),
                    Opnd::InsnOut(insn_id) => values[insn_id],
                }
            };
            // Handle control instructions first; they do not have a value
            if let Op::IfTrue { val, then_block, else_block } = op {
                match value_of(val.clone()) {
                    Type::Const(Value::Int(0)) => block_worklist.push_back(*else_block),
                    Type::Const(Value::Int(n)) => block_worklist.push_back(*then_block),
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
            // Now handle expression-like instructions
            let new_value = match op {
                Op::Add {v0, v1} => {
                    match (value_of(v0.clone()), value_of(v1.clone())) {
                        (Type::Const(Value::Int(l)), Type::Const(Value::Int(r))) => Type::Const(Value::Int(l+r)),
                        _ => Type::Top,
                    }
                }
                Op::Phi { ins } => {
                    // Only take into account operands coming from from reachable blocks
                    ins.iter().fold(Type::Top, |acc, (block_id, opnd)| if executable[*block_id] { meet(acc, value_of(opnd.clone())) } else { acc })
                }
                _ => todo!(),
            };
            if meet(old_value, new_value) != old_value {
                values[insn_id] = new_value;
                insn_worklist.extend(&uses[insn_id]);
            }
        }
        while let Some(block_id) = block_worklist.pop_front() {
            if !executable[block_id] {
                executable[block_id] = true;
                insn_worklist.extend(&prog.blocks[block_id].insns);
            }
        }
    }
    AnalysisResult { block_executable: executable, insn_type: values, insn_uses: uses }
}

fn compute_uses(prog: &mut Program) -> Vec<Vec<InsnId>> {
    // Map of instructions to instructions that use them
    // uses[A] = { B, C } means that B and C both use A in their operands
    let num_insns = prog.insns.len();
    let mut uses: Vec<HashSet<InsnId>> = Vec::with_capacity(num_insns);
    uses.resize(num_insns, HashSet::new());
    for (insn_id, insn) in prog.insns.iter().enumerate() {
        let Insn {op, .. } = insn;
        let mut mark_use = |user: InsnId, opnd: &Opnd| {
            match opnd {
                Opnd::InsnOut(used) => {
                    uses[*used].insert(user);
                }
                _ => {}
            }
        };
        match op {
            Op::Phi { ins } => {
                for (_, opnd) in ins {
                    mark_use(insn_id, opnd);
                }
            }
            Op::Add { v0, v1 } => {
                mark_use(insn_id, v0);
                mark_use(insn_id, v1);
            }
            Op::SendStatic { args, .. } => {
                for opnd in args {
                    mark_use(insn_id, opnd);
                }
            }
            Op::Return { val, .. } => {
                mark_use(insn_id, val);
            }
            Op::IfTrue { val, .. } => {
                mark_use(insn_id, val);
            }
            Op::Jump { .. } => {}
        }
    }
    uses.into_iter().map(|set| set.into_iter().collect()).collect()
}

// TODO: port this to Rust
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

fn gen_torture_test(num_funs: usize) -> Program
{
    let mut rng = LCG::new(1337);

    let callees = random_dag(&mut rng, num_funs, 1, 10);

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


            for callee_id in callees {




            }



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





    let mut prog = gen_torture_test(200);
    compute_uses(&mut prog);
    //sctp(&mut prog);
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
        let ret_id = prog.push_insn(block_id, Op::Return { val: Opnd::InsnOut(add_id), parent_fun: fun_id });
        let uses = compute_uses(&mut prog);
        assert_eq!(uses[add_id], vec![ret_id]);
        assert_eq!(uses[ret_id], vec![]);
    }

    #[test]
    fn test_add() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add0_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let add1_id = prog.push_insn(block_id, Op::Add { v0: Opnd::InsnOut(add0_id), v1: Opnd::InsnOut(add0_id) });
        let uses = compute_uses(&mut prog);
        assert_eq!(uses[add0_id], vec![add1_id]);
        assert_eq!(uses[add1_id], vec![]);
    }

    #[test]
    fn test_phi() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let phi_id = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::InsnOut(add_id))] });
        let uses = compute_uses(&mut prog);
        assert_eq!(uses[add_id], vec![phi_id]);
        assert_eq!(uses[phi_id], vec![]);
    }

    #[test]
    fn test_send() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let send_id = prog.push_insn(block_id, Op::SendStatic { target: 123, args: vec![Opnd::InsnOut(add_id)], ret_block: 123 });
        let uses = compute_uses(&mut prog);
        assert_eq!(uses[add_id], vec![send_id]);
        assert_eq!(uses[send_id], vec![]);
    }

    #[test]
    fn test_iftrue() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let iftrue_id = prog.push_insn(block_id, Op::IfTrue { val: Opnd::InsnOut(add_id), then_block: 3, else_block: 4 });
        let uses = compute_uses(&mut prog);
        assert_eq!(uses[add_id], vec![iftrue_id]);
        assert_eq!(uses[iftrue_id], vec![]);
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
        let add1_id = prog.push_insn(block_id, Op::Add { v0: Opnd::InsnOut(add0_id), v1: Opnd::Const(Value::Int(5)) });
        let result = sctp(&mut prog);
        assert_eq!(result.insn_type[add1_id], Type::Const(Value::Int(12)));
    }

    #[test]
    fn test_phi_same_const() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let phi_id = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::InsnOut(add_id)), (block_id, Opnd::Const(Value::Int(7)))] });
        let result = sctp(&mut prog);
        assert_eq!(result.insn_type[phi_id], Type::Const(Value::Int(7)));
    }

    #[test]
    fn test_phi_different_const() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let phi_id = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::InsnOut(add_id)), (block_id, Opnd::Const(Value::Int(8)))] });
        let result = sctp(&mut prog);
        assert_eq!(result.insn_type[phi_id], Type::Top);
    }

    #[test]
    fn test_iftrue_top() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let phi_id = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::Const(Value::Int(1))), (block_id, Opnd::Const(Value::Int(2)))] });
        let then_block = prog.new_block();
        let else_block = prog.new_block();
        let iftrue_id = prog.push_insn(block_id, Op::IfTrue { val: Opnd::InsnOut(phi_id), then_block, else_block });
        let result = sctp(&mut prog);
        assert_eq!(result.insn_type[phi_id], Type::Top);
        assert_eq!(result.block_executable[then_block], true);
        assert_eq!(result.block_executable[else_block], true);
    }

    #[test]
    fn test_iftrue_const_then() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let then_block = prog.new_block();
        let else_block = prog.new_block();
        let iftrue_id = prog.push_insn(block_id, Op::IfTrue { val: Opnd::Const(Value::Int(1)), then_block, else_block });
        let result = sctp(&mut prog);
        assert_eq!(result.block_executable[then_block], true);
        assert_eq!(result.block_executable[else_block], false);
    }

    #[test]
    fn test_iftrue_const_else() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let then_block = prog.new_block();
        let else_block = prog.new_block();
        let iftrue_id = prog.push_insn(block_id, Op::IfTrue { val: Opnd::Const(Value::Int(0)), then_block, else_block });
        let result = sctp(&mut prog);
        assert_eq!(result.block_executable[then_block], false);
        assert_eq!(result.block_executable[else_block], true);
    }
}
