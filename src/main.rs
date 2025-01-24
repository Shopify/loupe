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

    // Generate a random float between 0 and 1
    pub fn rand_float(&mut self) -> f64 {
        (self.next_u32() as f64) / (u32::MAX as f64)
    }

    // Sample an integer from a Pareto distribution with bounds
    // This version includes a maximum value to prevent extremely large numbers
    // alpha: shape parameter (must be positive)
    // min_value: scale parameter (must be positive)
    //
    // The alpha parameter controls the shape of the distribution:
    // - Lower alpha values (e.g., 1.1) produce more extreme values
    // - Higher alpha values (e.g., 3.0) produce values more concentrated near the minimum
    //
    pub fn pareto_int(&mut self, alpha: f64, min_value: u64, max_value: u64) -> u64 {
        assert!(alpha > 0.0, "Alpha must be positive");
        assert!(min_value > 0, "Minimum value must be positive");
        assert!(max_value > min_value, "Maximum value must be greater than minimum value");

        let u = self.rand_float();

        // Simpler bounded Pareto formula
        let min_f = min_value as f64;
        let max_f = max_value as f64;

        let l = min_f.powf(-alpha);
        let h = max_f.powf(-alpha);
        let x = (1.0 / ((h + u * (l - h)).powf(1.0/alpha))).round();

        x.clamp(min_value as f64, max_f) as u64
    }
}

#[derive(Default, Clone, PartialEq, Eq, Debug)]
pub struct Class {
    //name: String,

    // List of fields
    //ivars: Vec<String>,

    // Types associated with each field
    //ivar_types: Vec<Type>,

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

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ClassId(usize);
// TODO(max): Remove derive(Default) for FunId
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Default)]
pub struct FunId(usize);
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct BlockId(usize);
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct InsnId(usize);

impl std::fmt::Display for InsnId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

impl std::fmt::Display for FunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fn{}", self.0)
    }
}

// Type: Int, Nil, Class
#[derive(Debug, Clone, PartialEq)]
enum Type
{
    // Empty is the empty set (no info propagated yet or unreachable)
    Empty,
    Const(Value),
    Int,
    Bool,
    Object(HashSet<ClassId>),
    Any,
}

impl Type {
    fn object(class_id: ClassId) -> Type {
        Type::Object(HashSet::from([class_id]))
    }

    fn objects(class_ids: &Vec<ClassId>) -> Type {
        assert!(!class_ids.is_empty(), "Use Type::Empty instead");
        Type::Object(HashSet::from_iter(class_ids.iter().map(|id| *id)))
    }
}

fn union(left: &Type, right: &Type) -> Type {
    match (left, right) {
        (Type::Any, _) | (_, Type::Any) => Type::Any,
        (Type::Empty, x) | (x, Type::Empty) => x.clone(),
        (l, r) if l == r => l.clone(),
        // Int
        (Type::Int, Type::Const(Value::Int(_))) | (Type::Const(Value::Int(_)), Type::Int) => Type::Int,
        (Type::Const(Value::Int(_)), Type::Const(Value::Int(_))) => Type::Int,
        (Type::Int, Type::Int) => Type::Int,
        // Bool
        (Type::Bool, Type::Const(Value::Bool(_))) | (Type::Const(Value::Bool(_)), Type::Bool) => Type::Bool,
        (Type::Const(Value::Bool(_)), Type::Const(Value::Bool(_))) => Type::Bool,
        (Type::Bool, Type::Bool) => Type::Bool,
        // Object
        (Type::Object(l), Type::Object(r)) => Type::Object(l.union(r).map(|item| *item).collect()),
        // Other
        _ => Type::Any,
    }
}

// Home of our interprocedural CFG
#[derive(Default, Debug)]
struct Program
{
    classes: Vec<Class>,

    funs: Vec<Function>,

    blocks: Vec<Block>,

    // Set of blocks that contain a terminator instruction
    blocks_terminated: HashSet<BlockId>,

    insns: Vec<Insn>,

    // Main/entry function
    main: FunId,
}

impl Program {
    // Register a class and assign it an id
    pub fn new_class(&mut self) -> ClassId {
        let id = self.classes.len();
        self.classes.push(Class::default());
        ClassId(id)
    }

    // Register a method associated with a class
    pub fn new_method(&mut self, class_id: ClassId, name: String) -> (FunId, BlockId) {
        let (m_id, b_id) = self.new_fun();

        // Register the method with the given class
        let k = &mut self.classes[class_id.0];
        assert!(!k.methods.contains_key(&name));
        k.methods.insert(name, m_id);

        // Return method id and entry block id
        (m_id, b_id)
    }

    // Register a function and assign it an id
    pub fn new_fun(&mut self) -> (FunId, BlockId) {
        let id = FunId(self.funs.len());
        let entry_block = self.new_block(id);
        self.funs.push(Function { entry_block });
        (id, entry_block)
    }

    // Register a block and assign it an id
    pub fn new_block(&mut self, fun_id: FunId) -> BlockId {
        let id = BlockId(self.blocks.len());
        self.blocks.push(Block { fun_id, insns: vec![] });
        id
    }

    // Add an instruction to the program
    pub fn push_insn(&mut self, block: BlockId, op: Op) -> InsnId {
        // Check that we're not adding insns after a branch in an already terminated block
        if self.blocks_terminated.contains(&block) {
            panic!("Cannot push terminator instruction on block that is already terminated");
        }

        if op.is_terminator() {
            self.blocks_terminated.insert(block);
        }

        let insn = Insn {
            block_id: block,
            op,
        };
        let id = InsnId(self.insns.len());
        self.insns.push(insn);

        // Add the insn to the block
        self.blocks[block.0].insns.push(id);

        id
    }

    fn add_phi_arg(&mut self, insn_id: InsnId, block_id: BlockId, opnd: Opnd) {
        let insn = &mut self.insns[insn_id.0];
        match insn {
            Insn { op: Op::Phi { ins }, .. } => ins.push((block_id, opnd)),
            _ => panic!("Can't append phi arg to non-phi {:?}", insn)
        }
    }

    fn entry_of(&self, fun_id: FunId) -> BlockId {
        self.funs[fun_id.0].entry_block
    }

    fn lookup_method(&self, class_id: ClassId, method_name: &String) -> Option<FunId> {
        self.classes[class_id.0].methods.get(method_name).copied()
    }
}

#[derive(Debug)]
struct Function
{
    entry_block: BlockId,

    // We don't need an explicit list of blocks
}

#[derive(Debug)]
struct Block
{
    // Do we need to keep the phi nodes separate?

    fun_id: FunId,
    insns: Vec<InsnId>,
}

// Remove this if the only thing we store is op
#[derive(Debug)]
struct Insn
{
    block_id: BlockId,
    op: Op,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug, Copy)]
pub enum Value {
    Nil,
    Int(i64),
    Bool(bool),
    Fun(FunId),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
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

    // Create a new instance of the class
    New { class: ClassId },

    // Start with a static send (no dynamic lookup)
    // to get the basics of the analysis working
    SendStatic { target: FunId, args: Vec<Opnd> },

    // Dynamic dispatch to a method with a given name
    SendDynamic { method: String, self_val: Opnd, args: Vec<Opnd> },

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

impl Op {
    // Check whether this is a branch which terminates a block
    fn is_terminator(&self) -> bool {
        match self {
            Op::Return {..} => true,
            Op::IfTrue {..} => true,
            Op::Jump {..} => true,
            _ => false,
        }
    }
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

impl AnalysisResult {
    fn type_of(&self, insn_id: InsnId) -> Type {
        self.insn_type[insn_id.0].clone()
    }

    fn is_executable(&self, block_id: BlockId) -> bool {
        self.block_executable[block_id.0]
    }
}

// Sparse conditionall type propagation
#[inline(never)]
fn sctp(prog: &Program) -> AnalysisResult
{
    let insn_uses = compute_uses(prog);
    let num_blocks = prog.blocks.len();
    let mut executable: Vec<bool> = Vec::with_capacity(num_blocks);
    executable.resize(num_blocks, false);

    let num_insns = prog.insns.len();
    let mut types: Vec<Type> = Vec::with_capacity(num_insns);
    types.resize(num_insns, Type::Empty);

    // Map of functions to instructions that called them
    let num_funs = prog.funs.len();
    let mut called_by: Vec<HashSet<InsnId>> = Vec::with_capacity(num_funs);
    called_by.resize(num_funs, HashSet::new());

    // Map of InsnId -> operands that flow to that insn
    // Flow goes from send arguments to function parameters and from return values to send results
    let mut flows_to: Vec<HashSet<Opnd>> = Vec::with_capacity(num_insns);
    flows_to.resize(num_insns, HashSet::new());

    // Mark entry as executable
    let entry = prog.entry_of(prog.main);
    executable[entry.0] = true;

    // Map of FunId -> list of that function's return values
    // TODO(max): Maybe cache return type of each function
    let mut func_returns: Vec<Vec<Opnd>> = Vec::with_capacity(num_funs);
    func_returns.resize(num_funs, vec![]);

    for (insn_id, insn) in prog.insns.iter().enumerate() {
        match insn.op {
            Op::Return { val, parent_fun  } => func_returns[parent_fun.0].push(val),
            _ => {}
        }
    }

    // Work list of instructions or blocks
    let mut block_worklist: VecDeque<BlockId> = VecDeque::new();
    let mut insn_worklist: VecDeque<InsnId> = VecDeque::from(prog.blocks[entry.0].insns.clone());

    let mut itr_count = 0;

    while block_worklist.len() > 0 || insn_worklist.len() > 0
    {
        while let Some(insn_id) = insn_worklist.pop_front() {
            itr_count += 1;

            let Insn {op, block_id, ..} = &prog.insns[insn_id.0];
            let old_type = &types[insn_id.0];
            let fun_id = prog.blocks[block_id.0].fun_id;
            let type_of = |opnd: &Opnd| -> Type {
                match opnd {
                    Opnd::Const(v) => Type::Const(*v),
                    Opnd::Insn(insn_id) => types[insn_id.0].clone(),
                }
            };
            let is_insn_reachable = |insn_id: InsnId| -> bool {
                executable[prog.insns[insn_id.0].block_id.0]
            };
            // Handle control instructions first; they do not have a value
            if let Op::IfTrue { val, then_block, else_block } = op {
                match type_of(val) {
                    Type::Empty => {},
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
                if type_of(val) == Type::Empty { continue; }
                for send_insn in &called_by[parent_fun.0] {
                    assert!(matches!(prog.insns[send_insn.0].op, Op::SendStatic { .. } | Op::SendDynamic { .. }));
                    let old_type = &types[send_insn.0];
                    if union(old_type, &type_of(&val)) != *old_type {
                        insn_worklist.push_back(*send_insn);
                    }
                }
                continue;
            };
            // Now handle expression-like instructions
            let new_type = match op {
                Op::Add {v0, v1} => {
                    match (type_of(v0), type_of(v1)) {
                        (Type::Empty, _) | (_, Type::Empty) => Type::Empty,
                        (Type::Const(Value::Int(l)), Type::Const(Value::Int(r))) =>
                            match l.checked_add(r) {
                                Some(result) => Type::Const(Value::Int(result)),
                                _ => Type::Int,
                            }
                        (l, r) if union(&l, &r) == Type::Int => Type::Int,
                        _ => Type::Any,
                    }
                }
                Op::Mul {v0, v1} => {
                    match (type_of(v0), type_of(v1)) {
                        (Type::Empty, _) | (_, Type::Empty) => Type::Empty,
                        (Type::Const(Value::Int(l)), Type::Const(Value::Int(r))) =>
                            match l.checked_mul(r) {
                                Some(result) => Type::Const(Value::Int(result)),
                                _ => Type::Int,
                            }
                        (l, r) if union(&l, &r) == Type::Int => Type::Int,
                        _ => Type::Any,
                    }
                }
                Op::LessThan {v0, v1} => {
                    match (type_of(v0), type_of(v1)) {
                        (Type::Empty, _) | (_, Type::Empty) => Type::Empty,
                        (Type::Const(Value::Int(l)), Type::Const(Value::Int(r))) => Type::Const(Value::Bool(l<r)),
                        (l, r) if union(&l, &r) == Type::Int => Type::Bool,
                        _ => Type::Any,
                    }
                }
                Op::IsNil { v } => {
                    match type_of(v) {
                        Type::Empty => Type::Empty,
                        Type::Const(Value::Nil) => Type::Const(Value::Bool(true)),
                        Type::Const(_) | Type::Int | Type::Bool => Type::Const(Value::Bool(false)),
                        _ => Type::Bool,
                    }
                }
                Op::Phi { ins } => {
                    // Only take into account operands coming from from reachable blocks
                    ins.iter().fold(Type::Empty, |acc, (block_id, opnd)| if executable[block_id.0] { union(&acc, &type_of(opnd)) } else { acc })
                }
                Op::Param { idx, parent_fun } => {
                    // TODO(max): Pull from callers?
                    flows_to[insn_id.0].iter().fold(Type::Empty, |acc, opnd| union(&acc, &type_of(opnd)))
                }
                Op::SendStatic { target, args } => {
                    let target_entry_id = prog.entry_of(*target);
                    if called_by[target.0].insert(insn_id) {
                        // Newly inserted; enqueue target and update flow relation
                        block_worklist.push_back(target_entry_id);
                        // Flow arguments to parameters
                        // NOTE: assumes all Param are in the first block of a function
                        for target_insn in &prog.blocks[target_entry_id.0].insns {
                            match prog.insns[target_insn.0].op {
                                Op::Param { idx, .. } => {
                                    assert!(idx < args.len());
                                    flows_to[target_insn.0].insert(args[idx]);
                                }
                                _ => {}
                            }
                        }
                        // TODO(max): Mark all Insn operands as being used by Param?
                    }
                    // If we have any new information for the parameters, enqueue them
                    for target_insn in &prog.blocks[target_entry_id.0].insns {
                        match prog.insns[target_insn.0].op {
                            Op::Param { idx, .. } => {
                                assert!(idx < args.len());
                                let arg_type = type_of(&args[idx]);
                                let old_type = &types[target_insn.0];
                                if union(old_type, &arg_type) != *old_type {
                                    insn_worklist.push_back(*target_insn);
                                }
                            }
                            _ => {}
                        }
                    }
                    for val in &func_returns[target.0] {
                        flows_to[insn_id.0].insert(*val);
                    }
                    flows_to[insn_id.0].iter().fold(Type::Empty, |acc, opnd| union(&acc, &type_of(opnd)))
                }
                Op::New { class } => Type::object(*class),
                Op::SendDynamic { method, self_val, args } => {
                    match type_of(self_val) {
                        Type::Object(class_ids) => {
                            let targets = class_ids.iter().filter_map(|class_id| prog.lookup_method(*class_id, method));
                            for target in targets {
                                let target_entry_id = prog.entry_of(target);
                                if called_by[target.0].insert(insn_id) {
                                    // Newly inserted; enqueue target and update flow relation
                                    block_worklist.push_back(target_entry_id);
                                    // Flow arguments to parameters
                                    // NOTE: assumes all Param are in the first block of a function
                                    for target_insn in &prog.blocks[target_entry_id.0].insns {
                                        match prog.insns[target_insn.0].op {
                                            Op::Param { idx, .. } => {
                                                assert!(idx < args.len());
                                                flows_to[target_insn.0].insert(args[idx]);
                                            }
                                            _ => {}
                                        }
                                    }
                                    // TODO(max): Mark all Insn operands as being used by Param?
                                }
                                // If we have any new information for the parameters, enqueue them
                                for target_insn in &prog.blocks[target_entry_id.0].insns {
                                    match prog.insns[target_insn.0].op {
                                        Op::Param { idx, .. } => {
                                            assert!(idx < args.len());
                                            let arg_type = type_of(&args[idx]);
                                            let old_type = &types[target_insn.0];
                                            if union(old_type, &arg_type) != *old_type {
                                                insn_worklist.push_back(*target_insn);
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                                for val in &func_returns[target.0] {
                                    flows_to[insn_id.0].insert(*val);
                                }
                            }
                            flows_to[insn_id.0].iter().fold(Type::Empty, |acc, opnd| union(&acc, &type_of(opnd)))
                        }
                        ty => panic!("send to non-Object type {ty:?}"),
                    }
                }
                _ => todo!("op not yet supported {:?}", op),
            };
            if union(&old_type, &new_type) != *old_type {
                types[insn_id.0] = new_type;
                for use_id in &insn_uses[insn_id.0] {
                    if is_insn_reachable(*use_id) {
                        insn_worklist.push_back(*use_id);
                    }
                }
            }
        }

        while let Some(block_id) = block_worklist.pop_front() {
            itr_count += 1;

            if !executable[block_id.0] {
                executable[block_id.0] = true;
                insn_worklist.extend(&prog.blocks[block_id.0].insns);
            }
        }
    }

    AnalysisResult {
        block_executable: executable,
        insn_type: types,
        insn_uses,
        itr_count
    }
}

#[inline(never)]
// Map of InsnId -> instructions that use that insn
fn compute_uses(prog: &Program) -> Vec<Vec<InsnId>> {
    // Map of instructions to instructions that use them
    // uses[A] = { B, C } means that B and C both use A in their operands
    let num_insns = prog.insns.len();
    let mut uses: Vec<HashSet<InsnId>> = Vec::with_capacity(num_insns);
    uses.resize(num_insns, HashSet::new());
    for (insn_id, insn) in prog.insns.iter().enumerate() {
        let insn_id = InsnId(insn_id);
        let mut mark_use = |opnd: &Opnd| {
            match opnd {
                Opnd::Insn(used) => {
                    uses[used.0].insert(insn_id);
                }
                _ => {}
            }
        };
        match &insn.op {
            Op::Phi { ins } => {
                for (_, opnd) in ins {
                    mark_use(opnd);
                }
            }
            Op::Add { v0, v1 } | Op::Mul {v0, v1 } | Op::LessThan { v0, v1 } => {
                mark_use(v0);
                mark_use(v1);
            }
            Op::IsNil { v } => {
                mark_use(v);
            }
            Op::SendStatic { args, .. } => {
                for opnd in args {
                    mark_use(opnd);
                }
            }
            Op::SendDynamic { self_val, args, .. } => {
                mark_use(self_val);
                for opnd in args {
                    mark_use(opnd);
                }
            }
            Op::Return { val, parent_fun } => {
                mark_use(val);
            }
            Op::IfTrue { val, .. } => {
                mark_use(val);
            }
            Op::Param { .. } => {}
            Op::Jump { .. } => {}
            Op::New { .. } => {}
        }
    }
    uses.into_iter().map(|set| set.into_iter().collect()).collect()
}

// Generate a random acyclic call graph
fn random_dag(rng: &mut LCG, num_nodes: usize, min_parents: usize, max_parents: usize) -> Vec<Vec<usize>>
{
    let mut callees: Vec<Vec<usize>> = Vec::new();

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
    prog.main = FunId(0);

    // For each function to be generated
    for fun_id in 0..num_funs {
        let fun_id = FunId(fun_id);
        let (f_id, entry_block) = prog.new_fun();
        assert!(f_id == fun_id);

         // List of callees for this function
        let callees = &callees[fun_id.0];

        // If this is a leaf method
        if callees.is_empty() {
            // Return a constant
            let const_val = if rng.rand_bool() {
                Value::Nil
            } else {
                let rand_int = rng.rand_usize(1, 500) as i64;
                Value::Int(rand_int)
            };

            //println!("{}: {:?}", fun_id, const_val);

            prog.push_insn(
                entry_block,
                Op::Return { val: Opnd::Const(const_val), parent_fun: fun_id }
            );
        } else {
            //println!("{}: callees: {:?}", fun_id, callees);

            let mut last_block = entry_block;
            let mut sum_val = ZERO;

            // Call each callee
            for callee_id in callees {
                let nil_block = prog.new_block(fun_id);
                let int_block = prog.new_block(fun_id);
                let sum_block = prog.new_block(fun_id);

                let call_insn = prog.push_insn(
                    last_block,
                    Op::SendStatic { target: FunId(*callee_id), args: vec![] }
                );
                let isnil_insn = prog.push_insn(last_block, Op::IsNil { v: Opnd::Insn(call_insn) });
                prog.push_insn(last_block, Op::IfTrue { val: Opnd::Insn(isnil_insn), then_block: nil_block, else_block: int_block });

                // Both branches go to the sum block
                prog.push_insn(nil_block, Op::Jump { target: sum_block });
                prog.push_insn(int_block, Op::Jump { target: sum_block });

                // Compute the sum
                let phi_id = prog.push_insn(sum_block, Op::Phi { ins: vec![(nil_block, ZERO), (int_block, Opnd::Insn(call_insn))] });
                let add_id = prog.push_insn(sum_block, Op::Add { v0: sum_val.clone(), v1: Opnd::Insn(phi_id) });
                sum_val = Opnd::Insn(add_id);
                last_block = sum_block;
            }

            prog.push_insn(
                last_block,
                Op::Return { val: sum_val, parent_fun: fun_id }
            );
        }
    }

    prog
}

#[inline(never)]
fn gen_torture_test_2(num_classes: usize, num_roots: usize, dag_size: usize) -> Program
{
    const METHODS_PER_CLASS: usize = 10;

    let mut rng = LCG::new(1337);

    let mut prog = Program::default();

    // Generate a large number of classes
    let mut classes = Vec::new();
    for _ in 0..num_classes {
        let class_id = prog.new_class();
        classes.push(class_id);

        // Create methods for this class
        for j in 0..METHODS_PER_CLASS {
            let (m_id, entry_block) = prog.new_method(class_id, format!("m{}", j));

            // Return a random integer constant or nil
            let const_val = if rng.rand_bool() {
                let rand_int = rng.rand_usize(1, 500) as i64;
                Value::Int(rand_int)
            } else {
                Value::Nil
            };

            prog.push_insn(
                entry_block,
                Op::Return { val: Opnd::Const(const_val), parent_fun: m_id }
            );
        }
    }

    let (main_fun, main_entry) = prog.new_fun();
    prog.main = main_fun;

    // Create one instance of each class
    let mut objects = Vec::new();
    for class_id in classes {
        let obj = prog.push_insn(
            main_entry,
            Op::New { class: class_id }
        );
        objects.push(obj);
    }

    // For each root/subgraph
    for _ in 0..num_roots {
        // Generate a random call graph
        let callees = random_dag(&mut rng, dag_size, 1, 10);

        // Map of DAG node indices to function ids
        let mut fun_map = HashMap::new();

        // For each DAG node, going from leafs to root
        for (dag_idx, callees) in callees.into_iter().enumerate().rev() {
            let (fun_id, entry_block) = prog.new_fun();
            let param_id = prog.push_insn(entry_block, Op::Param { idx: 0, parent_fun: fun_id });

            // Map function ids to DAG node indices
            fun_map.insert(dag_idx, fun_id);

            // Call a random method on the object received as a parameter
            let m_no = rng.rand_usize(0, METHODS_PER_CLASS - 1);
            let m_name = format!("m{}", m_no);
            let call_insn = prog.push_insn(
                entry_block,
                Op::SendDynamic { method: m_name, self_val: Opnd::Insn(param_id) , args: vec![] }
            );

            let nil_block = prog.new_block(fun_id);
            let int_block = prog.new_block(fun_id);
            let sum_block = prog.new_block(fun_id);

            // Check if the method returned nil or an integer
            let isnil_insn = prog.push_insn(entry_block, Op::IsNil { v: Opnd::Insn(call_insn) });
            prog.push_insn(entry_block, Op::IfTrue { val: Opnd::Insn(isnil_insn), then_block: nil_block, else_block: int_block });

            // Both branches go to the sum block
            prog.push_insn(nil_block, Op::Jump { target: sum_block });
            prog.push_insn(int_block, Op::Jump { target: sum_block });

            // Compute the sum
            let phi_id = prog.push_insn(sum_block, Op::Phi { ins: vec![(nil_block, ZERO), (int_block, Opnd::Insn(call_insn))] });
            let mut sum_val = Opnd::Insn(phi_id);

            // Call each callee
            for callee_node_id in callees {
                // Call the function
                let callee_fun_id = fun_map.get(&callee_node_id).unwrap();
                let call_insn = prog.push_insn(
                    sum_block,
                    Op::SendStatic { target: *callee_fun_id, args: vec![Opnd::Insn(param_id)] }
                );

                // Add this value to the sum
                let add_id = prog.push_insn(sum_block, Op::Add { v0: sum_val.clone(), v1: Opnd::Insn(call_insn) });
                sum_val = Opnd::Insn(add_id);
            }

            prog.push_insn(
                sum_block,
                Op::Return { val: sum_val, parent_fun: fun_id }
            );
        }

        // In practice, most call sites are monomorphic.
        // Sizes should skew small most of the time but follow a power law
        let num_classes = rng.pareto_int(0.45, 1, min(150, num_classes as u64)) as usize;
        println!("num_classes: {}", num_classes);

        // Randomly select class instances
        let mut objs = HashSet::new();
        while objs.len() < num_classes {
            objs.insert(rng.choice(&objects));
        }

        // Call the new root function with each class instance
        let root_fun_id = fun_map.get(&0).unwrap();
        for obj in objs {
            let obj = prog.push_insn(
                main_entry,
                Op::SendStatic { target: *root_fun_id, args: vec![Opnd::Insn(*obj)] }
            );
        }
    }

    prog
}

fn print_prog(prog: &Program, result: Option<AnalysisResult>) {
    println!("Entry: {}", prog.main);
    for (fun_id, fun) in prog.funs.iter().enumerate() {
        let fun_id = FunId(fun_id);
        println!("fun {fun_id}:");
        for (block_id, block) in prog.blocks.iter().enumerate() {
            let block_id = BlockId(block_id);
            // We don't keep a map of Function->Vec<BlockId>
            if block.fun_id == fun_id {
                println!("  block {block_id}:");
                for insn_id in &block.insns {
                    let insn = &prog.insns[insn_id.0];
                    match result {
                        Some(ref result) if !insn.op.is_terminator() => {
                            let ty = result.type_of(*insn_id);
                            println!("    {insn_id}:{ty:?} = {insn:?}");
                        }
                        _ => {
                            println!("    {insn_id}: {insn:?}");
                        }
                    }
                }
            }
        }
    }
}

// Time the execution of a function, produce a value in milliseconds
fn time_exec_ms<F, T>(f: F) -> (T, f64)
where
    F: FnOnce() -> T
{
    use std::time::Instant;
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    let ms = duration.as_secs_f64() * 1000.0;
    (result, ms)
}

fn main()
{
    // TODO:
    // 1. Start with intraprocedural analysis
    // 2. Get interprocedural analysis working with direct send (no classes, no native methods)
    // 3. Once confident that interprocedural analysis is working, then add classes and objects



    let prog = gen_torture_test_2(5_000, 200, 750);
    //print_prog(&prog, Some(result));
    //let prog = gen_torture_test_2(2, 1, 2);

    let (result, time_ms) = time_exec_ms(|| sctp(&prog));


    // Check that all functions marked executable
    let mut exec_fn_count = 0;
    for fun in &prog.funs {
        let entry_id = fun.entry_block;
        if result.block_executable[entry_id.0] {
            exec_fn_count += 1;
        }
    }
    println!("exec_fn_count: {}", exec_fn_count);




    // Check that the main return type is integer
    for (insn_id, insn) in prog.insns.iter().enumerate() {
        if let Op::Return { val: Opnd::Insn(ret_id), parent_fun } = &insn.op {
            if *parent_fun == prog.main {
                let ret_type = &result.insn_type[ret_id.0];
                println!("{insn_id}: main return type: {:?}", ret_type);

                match ret_type {
                    Type::Int => {}
                    Type::Const(_) => {}
                    _ => panic!("output type should be an int but got {ret_type:?}"),
                }
            }
        }
    }

    println!("Total function count: {}", prog.funs.len());
    println!("Total instruction count: {}", prog.insns.len());
    println!("analysis time: {:.1} ms", time_ms);
    println!("itr count: {}", result.itr_count);
    println!();







    let prog = gen_torture_test(20_000);
    let (result, time_ms) = time_exec_ms(|| sctp(&prog));

    // Check that all functions marked executable
    for fun in &prog.funs {
        let entry_id = fun.entry_block;
        if !result.block_executable[entry_id.0] {
            panic!("all function entry blocks should be executable");
        }
    }

    // Check that the main return type is integer
    for (insn_id, insn) in prog.insns.iter().enumerate() {
        if let Op::Return { val: Opnd::Insn(ret_id), parent_fun } = &insn.op {
            if *parent_fun == prog.main {
                let ret_type = &result.insn_type[ret_id.0];
                println!("{insn_id}: main return type: {:?}", ret_type);

                match ret_type {
                    Type::Int => {}
                    Type::Const(_) => {}
                    _ => panic!("output type should be an int but got {ret_type:?}"),
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
        assert_eq!(union(&Type::Any, &Type::Any), Type::Any);
        assert_eq!(union(&Type::Any, &Type::Int), Type::Any);
        assert_eq!(union(&Type::Any, &Type::Empty), Type::Any);
        assert_eq!(union(&Type::Any, &Type::Int), Type::Any);
        assert_eq!(union(&Type::Any, &Type::Bool), Type::Any);
        assert_eq!(union(&Type::Any, &Type::Const(Value::Int(5))), Type::Any);
        assert_eq!(union(&Type::Any, &Type::Const(Value::Bool(true))), Type::Any);
    }

    #[test]
    fn test_empty() {
        assert_eq!(union(&Type::Empty, &Type::Any), Type::Any);
        assert_eq!(union(&Type::Empty, &Type::Int), Type::Int);
        assert_eq!(union(&Type::Empty, &Type::Empty), Type::Empty);
        assert_eq!(union(&Type::Empty, &Type::Const(Value::Int(5))), Type::Const(Value::Int(5)));
        assert_eq!(union(&Type::Empty, &Type::Int), Type::Int);
        assert_eq!(union(&Type::Empty, &Type::Bool), Type::Bool);
        assert_eq!(union(&Type::Empty, &Type::Const(Value::Bool(true))), Type::Const(Value::Bool(true)));
    }

    #[test]
    fn test_const() {
        assert_eq!(union(&Type::Const(Value::Int(3)), &Type::Const(Value::Int(3))), Type::Const(Value::Int(3)));
        assert_eq!(union(&Type::Const(Value::Bool(true)), &Type::Const(Value::Bool(true))), Type::Const(Value::Bool(true)));
        assert_eq!(union(&Type::Const(Value::Int(3)), &Type::Const(Value::Bool(true))), Type::Any);
    }

    #[test]
    fn test_type() {
        assert_eq!(union(&Type::Const(Value::Int(3)), &Type::Const(Value::Int(4))), Type::Int);
        assert_eq!(union(&Type::Const(Value::Int(3)), &Type::Int), Type::Int);

        assert_eq!(union(&Type::Const(Value::Bool(true)), &Type::Const(Value::Bool(false))), Type::Bool);
        assert_eq!(union(&Type::Const(Value::Bool(true)), &Type::Bool), Type::Bool);

        assert_eq!(union(&Type::Int, &Type::Bool), Type::Any);
    }

    #[test]
    fn test_object() {
        assert_eq!(union(&Type::object(ClassId(3)), &Type::Empty), Type::object(ClassId(3)));
        assert_eq!(union(&Type::object(ClassId(3)), &Type::object(ClassId(4))),
                   Type::objects(&vec![ClassId(3), ClassId(4)]));
        assert_eq!(union(&Type::objects(&vec![ClassId(3), ClassId(4)]), &Type::object(ClassId(5))),
                   Type::objects(&vec![ClassId(3), ClassId(4), ClassId(5)]));
        assert_eq!(union(&Type::object(ClassId(5)), &Type::objects(&vec![ClassId(3), ClassId(4)])),
                   Type::objects(&vec![ClassId(3), ClassId(4), ClassId(5)]));
    }
}

#[cfg(test)]
mod compute_uses_tests {
    use super::*;

    fn prog_with_empty_fun() -> (Program, FunId, BlockId) {
        let mut prog = Program::default();
        prog.main = FunId(0);
        let (fun_id, block_id) = prog.new_fun();
        (prog, fun_id, block_id)
    }

    #[test]
    fn test_return() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let ret_id = prog.push_insn(block_id, Op::Return { val: Opnd::Insn(add_id), parent_fun: fun_id });
        let insn_uses = compute_uses(&mut prog);
        assert_eq!(insn_uses[add_id.0], vec![ret_id]);
        assert_eq!(insn_uses[ret_id.0], vec![]);
    }

    #[test]
    fn test_add() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add0_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let add1_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Insn(add0_id), v1: Opnd::Insn(add0_id) });
        let insn_uses = compute_uses(&mut prog);
        assert_eq!(insn_uses[add0_id.0], vec![add1_id]);
        assert_eq!(insn_uses[add1_id.0], vec![]);
    }

    #[test]
    fn test_phi() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let phi_id = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::Insn(add_id))] });
        let insn_uses = compute_uses(&mut prog);
        assert_eq!(insn_uses[add_id.0], vec![phi_id]);
        assert_eq!(insn_uses[phi_id.0], vec![]);
    }

    #[test]
    fn test_send() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let (target, target_entry) = prog.new_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let send_id = prog.push_insn(block_id, Op::SendStatic { target, args: vec![Opnd::Insn(add_id)] });
        let insn_uses = compute_uses(&mut prog);
        assert_eq!(insn_uses[add_id.0], vec![send_id]);
        assert_eq!(insn_uses[send_id.0], vec![]);
    }

    #[test]
    fn test_iftrue() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let iftrue_id = prog.push_insn(block_id, Op::IfTrue { val: Opnd::Insn(add_id), then_block: BlockId(3), else_block: BlockId(4) });
        let insn_uses = compute_uses(&mut prog);
        assert_eq!(insn_uses[add_id.0], vec![iftrue_id]);
        assert_eq!(insn_uses[iftrue_id.0], vec![]);
    }
}

#[cfg(test)]
mod sctp_tests {
    use super::*;

    fn prog_with_empty_fun() -> (Program, FunId, BlockId) {
        let mut prog = Program::default();
        prog.main = FunId(0);
        let (fun_id, block_id) = prog.new_fun();
        (prog, fun_id, block_id)
    }

    #[test]
    fn test_add_const() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let result = sctp(&mut prog);
        assert_eq!(result.type_of(add_id), Type::Const(Value::Int(7)));
    }

    #[test]
    fn test_add_insn() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add0_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let add1_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Insn(add0_id), v1: Opnd::Const(Value::Int(5)) });
        let result = sctp(&mut prog);
        assert_eq!(result.type_of(add1_id), Type::Const(Value::Int(12)));
    }

    #[test]
    fn test_isnil_non_nil_const() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let isnil_id = prog.push_insn(block_id, Op::IsNil { v: Opnd::Const(Value::Int(3)) });
        let result = sctp(&mut prog);
        assert_eq!(result.type_of(isnil_id), Type::Const(Value::Bool(false)));
    }

    #[test]
    fn test_isnil_nil() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let isnil_id = prog.push_insn(block_id, Op::IsNil { v: NIL });
        let result = sctp(&mut prog);
        assert_eq!(result.type_of(isnil_id), Type::Const(Value::Bool(true)));
    }

    #[test]
    fn test_isnil_int_type() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let phi_id = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::Const(Value::Int(3))), (block_id, Opnd::Const(Value::Int(4)))] });
        let isnil_id = prog.push_insn(block_id, Op::IsNil { v: Opnd::Insn(phi_id) });
        let result = sctp(&mut prog);
        assert_eq!(result.type_of(phi_id), Type::Int);
        assert_eq!(result.type_of(isnil_id), Type::Const(Value::Bool(false)));
    }

    #[test]
    fn test_isnil_any() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let phi_id = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::Const(Value::Bool(true))), (block_id, Opnd::Const(Value::Int(4)))] });
        let isnil_id = prog.push_insn(block_id, Op::IsNil { v: Opnd::Insn(phi_id) });
        let result = sctp(&mut prog);
        assert_eq!(result.type_of(phi_id), Type::Any);
        assert_eq!(result.type_of(isnil_id), Type::Bool);
    }

    #[test]
    fn test_less_than() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let lt0_id = prog.push_insn(block_id, Op::LessThan { v0: Opnd::Const(Value::Int(7)), v1: Opnd::Const(Value::Int(8)) });
        let lt1_id = prog.push_insn(block_id, Op::LessThan { v0: Opnd::Const(Value::Int(8)), v1: Opnd::Const(Value::Int(8)) });
        let result = sctp(&mut prog);
        assert_eq!(result.type_of(lt0_id), Type::Const(Value::Bool(true)));
        assert_eq!(result.type_of(lt1_id), Type::Const(Value::Bool(false)));
    }

    #[test]
    fn test_phi_same_const() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let phi_id = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::Insn(add_id)), (block_id, Opnd::Const(Value::Int(7)))] });
        let result = sctp(&mut prog);
        assert_eq!(result.type_of(phi_id), Type::Const(Value::Int(7)));
    }

    #[test]
    fn test_phi_different_const() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let phi_id = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::Insn(add_id)), (block_id, Opnd::Const(Value::Int(8)))] });
        let result = sctp(&mut prog);
        assert_eq!(result.type_of(phi_id), Type::Int);
    }

    #[test]
    fn test_phi_different_type() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let add_id = prog.push_insn(block_id, Op::Add { v0: Opnd::Const(Value::Int(3)), v1: Opnd::Const(Value::Int(4)) });
        let phi_id = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::Insn(add_id)), (block_id, TRUE)] });
        let result = sctp(&mut prog);
        assert_eq!(result.type_of(phi_id), Type::Any);
    }

    #[test]
    fn test_iftrue_any() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let phi_id = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, ONE), (block_id, TWO)] });
        let then_block = prog.new_block(fun_id);
        let else_block = prog.new_block(fun_id);
        let iftrue_id = prog.push_insn(block_id, Op::IfTrue { val: Opnd::Insn(phi_id), then_block, else_block });
        let result = sctp(&mut prog);
        assert_eq!(result.type_of(phi_id), Type::Int);
        assert_eq!(result.is_executable(then_block), true);
        assert_eq!(result.is_executable(else_block), true);
    }

    #[test]
    fn test_iftrue_const_then() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let then_block = prog.new_block(fun_id);
        let else_block = prog.new_block(fun_id);
        let iftrue_id = prog.push_insn(block_id, Op::IfTrue { val: TRUE, then_block, else_block });
        let result = sctp(&mut prog);
        assert_eq!(result.is_executable(then_block), true);
        assert_eq!(result.is_executable(else_block), false);
    }

    #[test]
    fn test_iftrue_const_else() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let then_block = prog.new_block(fun_id);
        let else_block = prog.new_block(fun_id);
        let iftrue_id = prog.push_insn(block_id, Op::IfTrue { val: FALSE, then_block, else_block });
        let result = sctp(&mut prog);
        assert_eq!(result.is_executable(then_block), false);
        assert_eq!(result.is_executable(else_block), true);
    }

    #[test]
    fn test_jump() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let target = prog.new_block(fun_id);
        let iftrue_id = prog.push_insn(block_id, Op::Jump { target });
        let result = sctp(&mut prog);
        assert_eq!(result.is_executable(target), true);
    }

    #[test]
    fn test_one_return_flows_to_send() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let (target, target_entry) = prog.new_fun();
        let return_id = prog.push_insn(target_entry, Op::Return { val: Opnd::Const(Value::Int(5)), parent_fun: target });
        let send_id = prog.push_insn(block_id, Op::SendStatic { target, args: vec![] });
        let result = sctp(&mut prog);
        assert_eq!(result.is_executable(target_entry), true);
        assert_eq!(result.is_executable(block_id), true);
        assert_eq!(result.type_of(send_id), Type::Const(Value::Int(5)));
    }

    #[test]
    fn test_one_send_flows_to_param() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let (target, target_entry) = prog.new_fun();
        let param_id = prog.push_insn(target_entry, Op::Param { idx: 0, parent_fun: target });
        let return_id = prog.push_insn(target_entry, Op::Return { val: Opnd::Insn(param_id), parent_fun: target });
        let send_id = prog.push_insn(block_id, Op::SendStatic { target, args: vec![Opnd::Const(Value::Int(5))] });
        let result = sctp(&mut prog);
        assert_eq!(result.is_executable(target_entry), true);
        assert_eq!(result.type_of(param_id), Type::Const(Value::Int(5)));
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
        assert_eq!(result.is_executable(target_entry), true);
        assert_eq!(result.type_of(param_id), Type::Int);
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
        assert_eq!(result.is_executable(target_entry), true);
        assert_eq!(result.type_of(param0_id), Type::Const(Value::Int(3)));
        assert_eq!(result.type_of(param1_id), Type::Const(Value::Int(4)));
        assert_eq!(result.type_of(send_id), Type::Const(Value::Int(7)));
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
        let body_id = prog.new_block(fun_id);
        let end_id = prog.new_block(fun_id);

        prog.push_insn(entry_id, Op::Jump { target: body_id });
        let n = prog.push_insn(body_id, Op::Phi { ins: vec![(entry_id, ZERO)] });
        let n_inc = prog.push_insn(body_id, Op::Add { v0: Opnd::Insn(n), v1: ONE });
        prog.add_phi_arg(n, body_id, Opnd::Insn(n_inc));
        let cond = prog.push_insn(body_id, Op::LessThan { v0: Opnd::Insn(n), v1: Opnd::Const(Value::Int(100)) });
        prog.push_insn(body_id, Op::IfTrue { val: Opnd::Insn(cond), then_block: body_id, else_block: end_id });
        prog.push_insn(end_id, Op::Return { val: Opnd::Insn(n), parent_fun: fun_id });
        let result = sctp(&mut prog);

        assert_eq!(result.is_executable(entry_id), true);
        assert_eq!(result.is_executable(body_id), true);
        assert_eq!(result.is_executable(end_id), true);
        assert_eq!(result.type_of(n), Type::Int);
        assert_eq!(result.type_of(n_inc), Type::Int);
        assert_eq!(result.type_of(cond), Type::Bool);
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
        let early_exit_id = prog.new_block(fun_id);
        let do_mul_id = prog.new_block(fun_id);
        prog.push_insn(fact_entry, Op::IfTrue { val: Opnd::Insn(lt), then_block: early_exit_id, else_block: do_mul_id });
        prog.push_insn(early_exit_id, Op::Return { val: ONE, parent_fun: fact_id });
        let sub = prog.push_insn(do_mul_id, Op::Add { v0: Opnd::Insn(n), v1: Opnd::Const(Value::Int(-1)) });
        let rec = prog.push_insn(do_mul_id, Op::SendStatic { target: fact_id, args: vec![Opnd::Insn(sub)] });
        let mul = prog.push_insn(do_mul_id, Op::Mul { v0: Opnd::Insn(n), v1: Opnd::Insn(rec) });
        prog.push_insn(do_mul_id, Op::Return { val: Opnd::Insn(mul), parent_fun: fact_id });
        let result = sctp(&mut prog);
        assert_eq!(result.is_executable(entry_id), true);
        assert_eq!(result.is_executable(fact_entry), true);
        assert_eq!(result.is_executable(early_exit_id), true);
        assert_eq!(result.is_executable(do_mul_id), true);
        assert_eq!(result.type_of(outside_call), Type::Int);
        assert_eq!(result.type_of(n), Type::Int);
        assert_eq!(result.type_of(lt), Type::Bool);
        assert_eq!(result.type_of(sub), Type::Int);
        assert_eq!(result.type_of(rec), Type::Int);
        assert_eq!(result.type_of(mul), Type::Int);
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
        let (mut prog, fun_id, entry_id) = prog_with_empty_fun();
        let (fib_id, fib_entry) = prog.new_fun();
        let outside_call = prog.push_insn(entry_id, Op::SendStatic { target: fib_id, args: vec![Opnd::Const(Value::Int(100))] });
        let n = prog.push_insn(fib_entry, Op::Param { idx: 0, parent_fun: fib_id });
        let lt = prog.push_insn(fib_entry, Op::LessThan { v0: Opnd::Insn(n), v1: TWO });
        let early_exit_id = prog.new_block(fun_id);
        let do_add_id = prog.new_block(fun_id);
        prog.push_insn(fib_entry, Op::IfTrue { val: Opnd::Insn(lt), then_block: early_exit_id, else_block: do_add_id });
        prog.push_insn(early_exit_id, Op::Return { val: Opnd::Insn(n), parent_fun: fib_id });
        let sub1 = prog.push_insn(do_add_id, Op::Add { v0: Opnd::Insn(n), v1: Opnd::Const(Value::Int(-1)) });
        let rec1 = prog.push_insn(do_add_id, Op::SendStatic { target: fib_id, args: vec![Opnd::Insn(sub1)] });
        let sub2 = prog.push_insn(do_add_id, Op::Add { v0: Opnd::Insn(n), v1: Opnd::Const(Value::Int(-2)) });
        let rec2 = prog.push_insn(do_add_id, Op::SendStatic { target: fib_id, args: vec![Opnd::Insn(sub2)] });
        let add = prog.push_insn(do_add_id, Op::Add { v0: Opnd::Insn(rec1), v1: Opnd::Insn(rec2) });
        prog.push_insn(do_add_id, Op::Return { val: Opnd::Insn(add), parent_fun: fib_id });
        let result = sctp(&mut prog);
        assert_eq!(result.is_executable(entry_id), true);
        assert_eq!(result.is_executable(fib_entry), true);
        assert_eq!(result.is_executable(early_exit_id), true);
        assert_eq!(result.is_executable(do_add_id), true);
        assert_eq!(result.type_of(outside_call), Type::Int);
        assert_eq!(result.type_of(n), Type::Int);
        assert_eq!(result.type_of(lt), Type::Bool);
        assert_eq!(result.type_of(sub1), Type::Int);
        assert_eq!(result.type_of(rec1), Type::Int);
        assert_eq!(result.type_of(sub2), Type::Int);
        assert_eq!(result.type_of(rec2), Type::Int);
        assert_eq!(result.type_of(add), Type::Int);
    }

    #[test]
    fn test_eval_dead_insn_repro() {
        /*
        foo(x)
            entry:
                jump phi_block;

            dead_block:
                t = add x, 1; // no uses, produces the int type
                t2 = is_nil t; // uses the add, so should get queued too, evaluates to false
                iftrue t2 then phi_block; else dead_block; // gets queued because it uses t2, marks dead_block as executable

            phi_block:
                phi [entry: 0, dead_block: 1337]
                return phi;
        */
        let (mut prog, caller_fun_id, caller_entry_id) = prog_with_empty_fun();
        let (callee_fun_id, callee_entry_id) = prog.new_fun();
        let outside_call = prog.push_insn(caller_entry_id, Op::SendStatic { target: callee_fun_id, args: vec![Opnd::Const(Value::Int(100))] });
        let dead_block_id = prog.new_block(callee_fun_id);
        let phi_block_id = prog.new_block(callee_fun_id);
        let x = prog.push_insn(callee_entry_id, Op::Param { idx: 0, parent_fun: callee_fun_id });
        prog.push_insn(callee_entry_id, Op::Jump { target: phi_block_id });

        let t = prog.push_insn(dead_block_id, Op::Add { v0: Opnd::Insn(x), v1: Opnd::Const(Value::Int(1)) });
        let t2 = prog.push_insn(dead_block_id, Op::IsNil { v: Opnd::Insn(t) });
        prog.push_insn(dead_block_id, Op::IfTrue { val: Opnd::Insn(t2), then_block: phi_block_id, else_block: dead_block_id });

        let phi = prog.push_insn(phi_block_id, Op::Phi { ins: vec![(callee_entry_id, Opnd::Const(Value::Int(0))), (dead_block_id, Opnd::Const(Value::Int(1337)))] });
        prog.push_insn(phi_block_id, Op::Return { val: Opnd::Insn(phi), parent_fun: callee_fun_id });

        let result = sctp(&mut prog);
        assert_eq!(result.type_of(phi), Type::Const(Value::Int(0)));
        assert_eq!(result.type_of(outside_call), Type::Const(Value::Int(0)));
    }

    #[test]
    fn test_new() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let obj = prog.push_insn(block_id, Op::New { class: ClassId(3) });
        prog.push_insn(block_id, Op::Return { val: Opnd::Insn(obj), parent_fun: fun_id });
        let result = sctp(&mut prog);
        assert_eq!(result.type_of(obj), Type::object(ClassId(3)));
    }

    #[test]
    fn test_phi_new() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let obj3 = prog.push_insn(block_id, Op::New { class: ClassId(3) });
        let obj4 = prog.push_insn(block_id, Op::New { class: ClassId(4) });
        let phi = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::Insn(obj3)), (block_id, Opnd::Insn(obj4))] });
        prog.push_insn(block_id, Op::Return { val: Opnd::Insn(phi), parent_fun: fun_id });
        let result = sctp(&mut prog);
        assert_eq!(result.type_of(phi), Type::objects(&vec![ClassId(3), ClassId(4)]));

    }
}
