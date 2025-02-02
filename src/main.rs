#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use std::collections::{HashMap, HashSet, BTreeSet, VecDeque};
use std::cmp::{min, max};
use bit_set::BitSet;

// TODO(max): Figure out how to do a no-hash HashSet/HashMap for the various Id types floating
// around the program. We are doing a lot of needlessly expensive SipHash when we don't need DOS
// protection.

/// Produce an string representation of an integer with comma separator for thousands
pub fn int_str_grouped<Int: ToString>(n: Int) -> String
{
    let num_chars: Vec<char> = n.to_string().chars().rev().collect();

    let mut chars_sep = Vec::new();

    for idx in 0..num_chars.len() {
        chars_sep.push(num_chars[idx]);
        if (idx % 3) == 2 && idx < num_chars.len() - 1 {
            chars_sep.push(',');
        }
    }

    let num_str: String = chars_sep.into_iter().rev().collect();

    num_str
}

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
    pub fn rand_index(&mut self, max: usize) -> usize {
        (self.next_u32() as usize) % max
    }

    // Uniformly distributed usize in [min, max]
    pub fn rand_usize(&mut self, min: usize, max: usize) -> usize {
        assert!(max >= min);
        min + (self.next_u32() as usize) % (1 + max - min)
    }

    // Returns true with a specific percentage of probability
    pub fn pct_prob(&mut self, percent: usize) -> bool {
        let idx = self.rand_index(100);
        idx < percent
    }

    // Pick a random element from a slice
    pub fn choice<'a, T>(&mut self, slice: &'a [T]) -> &'a T {
        assert!(!slice.is_empty());
        &slice[self.rand_index(slice.len())]
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
        let x = (1.0 / ((h + u * (l - h)).powf(1.0 / alpha))).round();

        x.clamp(min_value as f64, max_f) as u64
    }
}

#[derive(Clone, Debug)]
pub struct Class {
    name: String,

    // List of fields
    ivars: Vec<String>,

    // List of methods
    methods: HashMap<String, FunId>,

    // KISS, ignore for now
    // Constructor method
    ctor: Option<FunId>,
}

impl std::fmt::Display for ClassId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Class@{}", self.0)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, PartialOrd, Ord)]
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
    Int,  // Special case of Object(INT_CLASS)
    Bool,  // Special case of Object(TRUE_CLASS, FALSE_CLASS)
    Object(BitSet),
    Any,
}

impl Type {
    fn object(class_id: ClassId) -> Type {
        let mut result = BitSet::with_capacity(class_id.0);
        result.insert(class_id.0);
        Type::Object(result)
    }

    fn two_objects(l: ClassId, r: ClassId) -> Type {
        let mut result = BitSet::with_capacity(if l.0 > r.0 { l.0 } else { r.0 });
        result.insert(l.0);
        result.insert(r.0);
        Type::Object(result)
    }

    fn objects(class_ids: &Vec<ClassId>) -> Type {
        assert!(!class_ids.is_empty(), "Use Type::Empty instead");
        let mut result = BitSet::with_capacity(class_ids.iter().map(|cid| cid.0).max().unwrap());
        for class_id in class_ids {
            result.insert(class_id.0);
        }
        Type::Object(result)
    }

    #[inline]
    fn class_id(&self) -> ClassId {
        match self {
            Type::Const(Value::Nil) => NIL_CLASS,
            Type::Const(Value::Int(_)) => INT_CLASS,
            Type::Int => INT_CLASS,
            Type::Const(Value::Bool(true)) => TRUE_CLASS,
            Type::Const(Value::Bool(false)) => FALSE_CLASS,
            _ => panic!("can't get single class_id of {self:?}")
        }
    }
}

fn union(left: &Type, right: &Type) -> Type {
    match (left, right) {
        (Type::Any, _) | (_, Type::Any) => Type::Any,
        (Type::Empty, x) | (x, Type::Empty) => x.clone(),
        (Type::Const(Value::Nil), Type::Const(Value::Nil)) => left.clone(),
        (Type::Const(Value::Int(l)), Type::Const(Value::Int(r))) if l == r => left.clone(),
        (Type::Const(Value::Int(l)), Type::Const(Value::Int(r))) => Type::Int,
        (Type::Const(Value::Int(_)), Type::Int) | (Type::Int, Type::Const(Value::Int(_))) => Type::Int,
        (Type::Int, Type::Int) => Type::Int,
        (Type::Const(Value::Bool(l)), Type::Const(Value::Bool(r))) if l == r => left.clone(),
        (Type::Const(Value::Bool(l)), Type::Const(Value::Bool(r))) => Type::Bool,
        (Type::Const(Value::Bool(_)), Type::Bool) | (Type::Bool, Type::Const(Value::Bool(_))) => Type::Bool,
        (Type::Bool, Type::Bool) => Type::Bool,
        (Type::Object(l), Type::Object(r)) => {
            let mut result = l.clone();
            result.union_with(r);
            Type::Object(result)
        }
        (Type::Bool, x) | (x, Type::Bool) => union(&Type::two_objects(TRUE_CLASS, FALSE_CLASS), x),
        (x, Type::Object(set)) | (Type::Object(set), x) => {
            let mut result = set.clone();
            result.insert(x.class_id().0);
            Type::Object(result)
        }
        (l, r) => Type::two_objects(l.class_id(), r.class_id())
    }
}

// Home of our interprocedural CFG
#[derive(Debug)]
struct Program
{
    classes: Vec<Class>,

    funs: Vec<Function>,

    blocks: Vec<Block>,

    insns: Vec<Insn>,

    // Main/entry function
    main: FunId,
}

impl Default for Program {
    fn default() -> Program {
        let mut result = Program { classes: vec![], funs: vec![], blocks: vec![], insns: vec![], main: FunId(0) };
        let c = result.new_class();
        assert_eq!(c, NIL_CLASS);
        let c = result.new_class();
        assert_eq!(c, INT_CLASS);
        let c = result.new_class();
        assert_eq!(c, TRUE_CLASS);
        let c = result.new_class();
        assert_eq!(c, FALSE_CLASS);
        result
    }
}

impl Program {
    // Register a class and assign it an id
    pub fn new_class(&mut self) -> ClassId {
        let id = self.classes.len();
        self.classes.push(Class {
            name: format!("$cls{id}"),
            ivars: Default::default(),
            methods: Default::default(),
            ctor: None
        });
        ClassId(id)
    }

    // Register a class and assign it an id
    pub fn new_class_with_ctor(&mut self, name: String) -> (ClassId, (FunId, BlockId)) {
        let id = self.classes.len();
        let ctor = self.new_fun();
        self.classes.push(Class {
            name,
            ivars: Default::default(),
            methods: HashMap::from([ ("initialize".into(), ctor.0) ]),
            ctor: Some(ctor.0)
        });
        (ClassId(id), ctor)
    }

    // Register a new ivar
    fn push_ivar(&mut self, class: ClassId, name: String) {
        assert!(!self.classes[class.0].ivars.contains(&name));
        self.classes[class.0].ivars.push(name.clone());
    }

    // Register a method associated with a class
    pub fn new_method(&mut self, class_id: ClassId, name: String) -> (FunId, BlockId) {
        let (m_id, b_id) = self.new_fun_with_name(name.clone());

        // Register the method with the given class
        let k = &mut self.classes[class_id.0];
        assert!(!k.methods.contains_key(&name));
        k.methods.insert(name, m_id);

        // Return method id and entry block id
        (m_id, b_id)
    }

    // Register a named function and assign it an id
    pub fn new_fun_with_name(&mut self, name: String) -> (FunId, BlockId) {
        let id = FunId(self.funs.len());
        let entry_block = self.new_block(id);
        self.funs.push(Function { name, entry_block });
        (id, entry_block)
    }

    // Register a function and assign it an id
    pub fn new_fun(&mut self) -> (FunId, BlockId) {
        self.new_fun_with_name("".to_string())
    }

    // Register a block and assign it an id
    pub fn new_block(&mut self, fun_id: FunId) -> BlockId {
        let id = BlockId(self.blocks.len());
        self.blocks.push(Block { fun_id, insns: vec![] });
        id
    }

    fn block_is_terminated(&self, block: BlockId) -> bool {
        match self.blocks[block.0].insns.last() {
            Some(insn_id) if self.insns[insn_id.0].op.is_terminator() => true,
            _ => false,
        }
    }

    // Add an instruction to the program
    pub fn push_insn(&mut self, block: BlockId, op: Op) -> InsnId {
        // Check that we're not adding insns after a branch in an already terminated block
        if self.block_is_terminated(block) {
            panic!("Cannot push terminator instruction on block that is already terminated");
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

    fn create_instance(&mut self, block_id: BlockId, class_id: ClassId) -> InsnId {
        let obj = self.push_insn(block_id, Op::New { class: class_id });
        match self.classes[class_id.0].ctor {
            Some(fun_id) => {
                self.push_insn(block_id, Op::SendDynamic { method: "initialize".into(), self_val: Opnd::Insn(obj), args: vec![] });
            }
            _ => {}
        }
        obj
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

    fn fun_containing(&self, insn_id: InsnId) -> FunId {
        self.blocks[self.insns[insn_id.0].block_id.0].fun_id
    }

    fn lookup_method(&self, class_id: ClassId, method_name: &String) -> Option<FunId> {
        self.classes[class_id.0].methods.get(method_name).copied()
    }

    pub fn rpo(&self, fun_id: FunId) -> Vec<BlockId> {
        self.rpo_from(self.funs[fun_id.0].entry_block)
    }

    fn rpo_from(&self, block: BlockId) -> Vec<BlockId> {
        let mut result = vec![];
        let mut visited = HashSet::new();
        self.po_traverse_from(block, &mut result, &mut visited);
        result.reverse();
        result
    }

    fn po_traverse_from(
        &self,
        block: BlockId,
        result: &mut Vec<BlockId>,
        visited: &mut HashSet<BlockId>,
    ) {
        visited.insert(block);
        match &self.insns[self.blocks[block.0].insns.last().unwrap().0].op {
            Op::Return { .. } => (),
            Op::IfTrue { then_block, else_block, .. } => {
                if !visited.contains(&then_block) {
                    self.po_traverse_from(*then_block, result, visited);
                }
                if !visited.contains(&else_block) {
                    self.po_traverse_from(*else_block, result, visited);
                }
            }
            Op::Jump { target } => {
                if !visited.contains(&target) {
                    self.po_traverse_from(*target, result, visited);
                }
            }
            insn => {
                panic!("Invalid terminator {insn:?}")
            }
        }
        result.push(block);
    }
}

#[derive(Debug)]
struct Function
{
    name: String,

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
#[derive(Debug, PartialEq)]
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
const NIL_CLASS: ClassId = ClassId(0);
const INT_CLASS: ClassId = ClassId(1);
const TRUE_CLASS: ClassId = ClassId(2);
const FALSE_CLASS: ClassId = ClassId(3);

#[derive(Debug, PartialEq)]
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
    Return { val: Opnd },

    // Load self parameter
    SelfParam { class_id: ClassId },
    // Load a function parameter.
    Param { idx: usize },

    // Get/set ivar values
    SetIvar { name: String, self_val: Opnd, val: Opnd },
    GetIvar { name: String, self_val: Opnd, },

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

    // Map of insn -> set of instructions that call the function containing insns
    called_by: Vec<HashSet<InsnId>>,
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

    // Analyze constructors of every class to determine which ivars are definitely initialized at
    // the end of the constructor.
    // Map of ClassId->bitset of definitely initialized ivars. Used jointly with ivar types.
    let ivar_initialized: Vec<SmallBitSet> = (0..prog.classes.len()).map(|class_id|
        analyze_ctor(prog, ClassId(class_id))
    ).collect();
    // TODO(max): Set all uninitialized ivars to nil in ivar_types

    // Map of ClassId->ivar types. Used jointly with ivar initialization bitsets.
    let num_classes = prog.classes.len();
    let mut ivar_types: Vec<HashMap<String, Type>> = Vec::with_capacity(num_classes);
    ivar_types.resize(num_classes, HashMap::new());
    for (class_id, class) in prog.classes.iter().enumerate() {
        for ivar in &class.ivars {
            ivar_types[class_id].insert(ivar.clone(), Type::Empty);
        }
    }

    let mut getivars: HashMap<(ClassId, &String), HashSet<InsnId>> = HashMap::new();

    for (insn_id, insn) in prog.insns.iter().enumerate() {
        match insn {
            Insn { block_id, op: Op::Return { val } } => {
                let parent_fun = prog.blocks[block_id.0].fun_id;
                func_returns[parent_fun.0].push(*val)
            }
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
            let type_of = |opnd: &Opnd| -> Type {
                match opnd {
                    Opnd::Const(v) => Type::Const(*v),
                    // TODO(max): Figure out how to stop cloning here
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
            if let Op::Return { val } = op {
                if type_of(val) == Type::Empty { continue; }
                let fun_id = prog.blocks[block_id.0].fun_id;
                for send_insn in &called_by[fun_id.0] {
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
                        // TODO(max): Add and use is_int()
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
                        // TODO(max): Add and use is_int()
                        (l, r) if union(&l, &r) == Type::Int => Type::Int,
                        _ => Type::Any,
                    }
                }
                Op::LessThan {v0, v1} => {
                    match (type_of(v0), type_of(v1)) {
                        (Type::Empty, _) | (_, Type::Empty) => Type::Empty,
                        (Type::Const(Value::Int(l)), Type::Const(Value::Int(r))) => Type::Const(Value::Bool(l<r)),
                        // TODO(max): Add and use is_int()
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
                Op::SelfParam { class_id } => Type::object(*class_id),
                Op::Phi { ins } => {
                    // Only take into account operands coming from from reachable blocks
                    ins.iter().fold(Type::Empty, |acc, (block_id, opnd)| if executable[block_id.0] { union(&acc, &type_of(opnd)) } else { acc })
                }
                Op::Param { idx } => {
                    flows_to[insn_id.0].iter().fold(Type::Empty, |acc, opnd| union(&acc, &type_of(opnd)))
                }
                Op::GetIvar { self_val, name } => {
                    let result = match type_of(self_val) {
                        Type::Object(classes) => {
                            for class_id in classes.iter() {
                                getivars.entry((ClassId(class_id), name)).or_insert(HashSet::new()).insert(insn_id);
                            }
                            classes.iter().fold(Type::Empty, |acc, class_id| union(&acc, &ivar_types[class_id][name]))
                        }
                        ty => panic!("getivar on non-Object type {ty:?}"),
                    };
                    result
                }
                Op::SetIvar { self_val, name, val } => {
                    // TODO(max): Somehow get a flows_to relationship here so the set enqueues the get later
                    match type_of(self_val) {
                        Type::Object(classes) => {
                            let val_ty = type_of(val);
                            for class_id in classes.iter() {
                                let mut old_type = ivar_types[class_id].get_mut(name).unwrap();
                                let union_type = union(old_type, &val_ty);
                                if union_type != *old_type {
                                    *old_type = union_type;
                                    match getivars.get(&(ClassId(class_id), name)) {
                                        Some(insns) => {
                                            for getivar in insns.iter() {
                                                insn_worklist.push_back(*getivar);
                                            }
                                        }
                                        None => {}
                                    }
                                }
                            }
                        }
                        ty => panic!("setivar on non-Object type {ty:?}"),
                    }
                    Type::Empty
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
                        for val in &func_returns[target.0] {
                            flows_to[insn_id.0].insert(*val);
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
                                // TODO(max): Make some shortcuts for checking if union(old, new) != old
                                // For example, something like if new > old, this is an easy yes
                                if union(old_type, &arg_type) != *old_type {
                                    insn_worklist.push_back(*target_insn);
                                }
                            }
                            Op::SelfParam { .. } => panic!("no self parameter allowed in static send target"),
                            _ => {}
                        }
                    }
                    flows_to[insn_id.0].iter().fold(Type::Empty, |acc, opnd| union(&acc, &type_of(opnd)))
                }
                Op::New { class } => Type::object(*class),
                Op::SendDynamic { method, self_val, args } => {
                    match type_of(self_val) {
                        Type::Object(class_ids) => {
                            let targets = class_ids.iter().filter_map(|class_id| {
                                let class_id = ClassId(class_id);
                                match prog.lookup_method(class_id, method) {
                                    Some(fun_id) => Some((class_id, fun_id)),
                                    _ => panic!("Invalid send of {method} to {class_id:?}"),
                                }
                            });
                            for (class_id, target) in targets {
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
                                    for val in &func_returns[target.0] {
                                        flows_to[insn_id.0].insert(*val);
                                    }
                                }
                                // If we have any new information for the parameters, enqueue them
                                for target_insn in &prog.blocks[target_entry_id.0].insns {
                                    match prog.insns[target_insn.0].op {
                                        Op::Param { idx, .. } => {
                                            assert!(idx < args.len());
                                            let arg_type = type_of(&args[idx]);
                                            let old_type = &types[target_insn.0];
                                            // TODO(max): Make some shortcuts for checking if union(old, new) != old
                                            // For example, something like if new > old, this is an easy yes
                                            if union(old_type, &arg_type) != *old_type {
                                                insn_worklist.push_back(*target_insn);
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            flows_to[insn_id.0].iter().fold(Type::Empty, |acc, opnd| union(&acc, &type_of(opnd)))
                        }
                        ty => panic!("send to non-Object type {ty:?}"),
                    }
                }
                _ => todo!("op not yet supported {:?}", op),
            };
            // TODO(max): Make some shortcuts for checking if union(old, new) != old
            // For example, something like if new > old, this is an easy yes
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
        itr_count,
        called_by
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
            Op::Return { val } => {
                mark_use(val);
            }
            Op::IfTrue { val, .. } => {
                mark_use(val);
            }
            Op::SelfParam { .. } => {}
            Op::Param { .. } => {}
            Op::Jump { .. } => {}
            Op::New { .. } => {}

            Op::SetIvar { self_val, val, .. } => {
                mark_use(self_val);
                mark_use(val);
            }
            Op::GetIvar { self_val, .. } => {
                mark_use(self_val);
            }
        }
    }
    uses.into_iter().map(|set| set.into_iter().collect()).collect()
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct SmallBitSet(usize);

impl SmallBitSet {
    fn all_ones() -> SmallBitSet {
        SmallBitSet(std::usize::MAX)
    }

    fn set(&mut self, idx: usize) {
        self.0 |= 1 << idx;
    }

    fn and(&self, other: SmallBitSet) -> SmallBitSet {
        SmallBitSet(self.0 & other.0)
    }
}

fn analyze_ctor(prog: &Program, class: ClassId) -> SmallBitSet {
    let class = &prog.classes[class.0];
    let ctor = match class.ctor {
        Some(ctor) => ctor,
        None => { return SmallBitSet(0); }
    };
    let mut self_param = None;
    // Find self parameter
    let entry = prog.funs[ctor.0].entry_block;
    for insn_id in &prog.blocks[entry.0].insns {
        match &prog.insns[insn_id.0].op {
            Op::SelfParam { .. } => {
                self_param = Some(insn_id);
                break;
            }
            _ => {}
        }
    }
    let self_param = match self_param {
        Some(insn_id) => insn_id,
        None => panic!("can't analyze ctor without SelfParam"),
    };
    let blocks = prog.rpo(ctor);
    let mut preds: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();
    for block in &blocks {
        preds.insert(*block, HashSet::new());
    }
    let mut block_out: HashMap<BlockId, SmallBitSet> = HashMap::new();
    for block in &blocks {
        block_out.insert(*block, SmallBitSet(0));
    }
    // Do abstract interpretation to flow definite ivar assignment
    loop {
        let mut changed = false;
        for block in &blocks {
            // Entrypoint does not have any preds so in that special case we give it SmallBitSet(0)
            let mut state =  preds[&block].iter().map(|block| block_out[block]).reduce(|acc, out| acc.and(out)).unwrap_or(SmallBitSet(0));
            for insn_id in &prog.blocks[block.0].insns {
                match &prog.insns[insn_id.0].op {
                    Op::IfTrue { then_block, else_block, .. } => {
                        preds.get_mut(then_block).unwrap().insert(*block);
                        preds.get_mut(else_block).unwrap().insert(*block);
                    }
                    Op::Jump { target } => {
                        preds.get_mut(target).unwrap().insert(*block);
                    }
                    Op::SetIvar { name, self_val, val } if *self_val == Opnd::Insn(*self_param) => {
                        let idx = match class.ivars.iter().position(|ivar| *ivar == *name) {
                            // TODO(max): Check if index is too big to represent in SmallBitSet
                            Some(idx) => idx,
                            None => panic!("Unknown ivar {name}"),
                        };
                        state.set(idx);
                    }
                    _ => {}
                }
            }
            if block_out[block] != state {
                block_out.insert(*block, state);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // Summarize the definite assignment by and-ing all the Return states together
    let mut result = SmallBitSet::all_ones();
    for block in &blocks {
        match &prog.insns[prog.blocks[block.0].insns.last().unwrap().0].op {
            Op::Return { .. } => {
                result = result.and(block_out[block]);
            }
            _ => {}
        }
    }
    result
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
                Op::Return { val: Opnd::Const(const_val) }
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
                Op::Return { val: sum_val }
            );
        }
    }

    prog
}

#[inline(never)]
fn gen_torture_test_2(num_classes: usize, num_roots: usize, dag_size: usize) -> Program
{
    const METHODS_PER_CLASS: usize = 10;
    const IVARS_PER_CLASS: usize = 10;

    let mut rng = LCG::new(1337);

    let mut prog = Program::default();

    // Generate a large number of classes
    let mut classes = Vec::new();
    for class_idx in 0..num_classes {
        let (class_id, (ctor_id, ctor_entry)) = prog.new_class_with_ctor(format!("class_{class_idx}"));
        classes.push(class_id);

        // Set up a constructor for this class
        {
            let self_id = prog.push_insn(ctor_entry, Op::SelfParam { class_id });

            // Create some ivars for this class
            for j in 0..IVARS_PER_CLASS {
                let ivar_name = format!("ivar_{}", j);
                prog.push_ivar(class_id, ivar_name.clone());

                // With some probability, initialize the ivar to a random integer
                // Some ivsrs are left uninitialized
                if rng.pct_prob(80) {
                    let val = Opnd::Const(Value::Int(rng.rand_usize(1, 7) as i64));
                    prog.push_insn(ctor_entry, Op::SetIvar { name: ivar_name, self_val: Opnd::Insn(self_id), val });
                }
            }

            // Constructor returns nil
            prog.push_insn(ctor_entry, Op::Return { val: NIL });
        }

        // Create methods for this class
        for j in 0..METHODS_PER_CLASS {
            let (m_id, entry_block) = prog.new_method(class_id, format!("m{}", j));

            // TODO: implement increment op with setivar as well?
            // 25% of the time

            let ret_val = if rng.rand_bool() {
                // Choose a random ivar
                let self_id = prog.push_insn(entry_block, Op::SelfParam { class_id });
                let ivar_idx = rng.rand_index(IVARS_PER_CLASS);
                let ivar_name = format!("ivar_{}", ivar_idx);
                let getivar_id = prog.push_insn(entry_block, Op::GetIvar { name: ivar_name, self_val: Opnd::Insn(self_id) });
                Opnd::Insn(getivar_id)
            } else if rng.rand_bool() {
                let rand_int = rng.rand_usize(1, 500) as i64;
                Opnd::Const(Value::Int(rand_int))
            } else {
                Opnd::Const(Value::Nil)
            };

            prog.push_insn(entry_block, Op::Return { val: ret_val });
        }
    }

    let (main_fun, main_entry) = prog.new_fun();
    prog.main = main_fun;

    // Create one instance of each class
    let mut objects = Vec::new();
    for class_id in classes {
        let obj = prog.create_instance(main_entry, class_id);
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
            let param_id = prog.push_insn(entry_block, Op::Param { idx: 0 });

            // Map function ids to DAG node indices
            fun_map.insert(dag_idx, fun_id);

            // Call a random method on the object received as a parameter
            let m_no = rng.rand_usize(0, METHODS_PER_CLASS - 1);
            let m_name = format!("m{}", m_no);
            let call_insn = prog.push_insn(
                entry_block,
                Op::SendDynamic { method: m_name, self_val: Opnd::Insn(param_id), args: vec![] }
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
                Op::Return { val: sum_val }
            );
        }

        // In practice, most call sites are monomorphic.
        // Sizes should skew small most of the time but follow a power law
        let num_classes = rng.pareto_int(0.45, 1, min(150, num_classes as u64)) as usize;
        // println!("num_classes: {}", num_classes);

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

fn print_prog(prog: &Program, result: Option<&AnalysisResult>) {
    for (class_id, class) in prog.classes.iter().enumerate() {
        let class_id = ClassId(class_id);
        println!("class {class_id}");
        for (ivar_idx, ivar_name) in class.ivars.iter().enumerate() {
            println!("  ivar {ivar_idx}: {ivar_name}");
        }
        for (method_name, fun_id) in class.methods.iter() {
            println!("  method {method_name}: {fun_id}");
        }
        println!("end");
    }
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
    let prog = gen_torture_test_2(5_000, 200, 750);
    //let prog = gen_torture_test_2(500, 20, 1);

    let (result, time_ms) = time_exec_ms(|| sctp(&prog));

    // Check that all functions marked executable
    let mut exec_fn_count = 0;
    for fun in &prog.funs {
        let entry_id = fun.entry_block;
        if result.block_executable[entry_id.0] {
            exec_fn_count += 1;
        }
    }
    println!("exec_fn_count: {}", int_str_grouped(exec_fn_count));
    let mut max_num_classes = 0;
    let mut max_insn_idx = 0;
    for (insn_idx, ty) in result.insn_type.iter().enumerate() {
        match ty {
            Type::Object(classes) => {
                if classes.len() > max_num_classes {
                    max_num_classes = classes.len();
                    max_insn_idx = insn_idx;
                }
            }
            _ => {}
        }
    }
    println!("max_num_classes: {}, for insn {:?}", int_str_grouped(max_num_classes), prog.insns[max_insn_idx]);

    // Check that the main return type is integer
    for (insn_id, insn) in prog.insns.iter().enumerate() {
        if let Insn { block_id, op: Op::Return { val: Opnd::Insn(ret_id) } } = &insn {
            let parent_fun = prog.blocks[block_id.0].fun_id;
            if parent_fun == prog.main {
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

    println!("Total function count: {}", int_str_grouped(prog.funs.len()));
    println!("Total instruction count: {}", int_str_grouped(prog.insns.len()));
    println!("itr count: {}", int_str_grouped(result.itr_count));
    println!("analysis time: {:.1} ms", time_ms);
    println!();

    // Check that all global functions (but not methods) are marked executable
    for (fun_id, fun) in prog.funs.iter().enumerate() {
        let entry_id = fun.entry_block;

        if !fun.name.starts_with("m") && !result.block_executable[entry_id.0] {
            panic!("function {fun_id} entry block not marked executable: {:?}", fun);
        }
    }

    //print_prog(&prog, Some(result));







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
        if let Insn { block_id, op: Op::Return { val: Opnd::Insn(ret_id) } } = &insn {
            let parent_fun = prog.blocks[block_id.0].fun_id;
            if parent_fun == prog.main {
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

    println!("itr count: {}", int_str_grouped(result.itr_count));
    println!("analysis time: {:.1} ms", time_ms);
}

#[derive(Debug, PartialEq, Clone)]
enum Token {
    Def,
    Class,
    End,
    Return,
    If,
    Else,
    Int(i64),
    Ident(String),
    LParen,
    RParen,
    Comma,
    Plus,
    Minus,
    Mul,
    Div,
    Equal,
    Dot,
}

#[derive(PartialEq)]
enum Assoc {
    Any,
    Left,
    Right,
}

struct Lexer<'a> {
    input: std::iter::Peekable<std::str::Chars<'a>>,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Lexer<'a> {
        Lexer { input: source.chars().peekable() }
    }

    fn read_word(&mut self, start: char) -> Token {
        let mut result: String = start.into();
        while let Some(c) = self.input.peek() {
            if c.is_alphabetic() {
                result.push(*c);
                self.input.next();
            } else {
                break;
            }
        }
        if result == "def" {
            Token::Def
        } else if result == "end" {
            Token::End
        } else if result == "return" {
            Token::Return
        } else if result == "if" {
            Token::If
        } else if result == "else" {
            Token::Else
        } else {
            Token::Ident(result)
        }
    }

    fn read_int(&mut self, start: char) -> Token {
        let mut result: i64 = start.to_digit(RADIX).unwrap().into();
        loop {
            match self.input.peek() {
                Some(c) if c.is_digit(RADIX) => {
                    let digit: i64 = c.to_digit(RADIX).unwrap().into();
                    self.input.next();
                    result = result * (RADIX as i64) + digit;
                }
                _ => { break; }
            }
        }
        Token::Int(result)
    }
}

const RADIX: u32 = 10;

impl<'a> Iterator for Lexer<'a> {
    type Item = Token;
    fn next(&mut self) -> Option<Token> {
        let c = self.input.find(|c| !c.is_whitespace());
        if c.is_none() { return None; }
        let c = c.unwrap();
        if c.is_alphabetic() {
            Some(self.read_word(c))
        } else if c.is_digit(RADIX) {
            Some(self.read_int(c))
        } else if c == '(' {
            Some(Token::LParen)
        } else if c == ')' {
            Some(Token::RParen)
        } else if c == ',' {
            Some(Token::Comma)
        } else if c == '+' {
            Some(Token::Plus)
        } else if c == '*' {
            Some(Token::Mul)
        } else if c == '=' {
            Some(Token::Equal)
        } else if c == '.' {
            Some(Token::Dot)
        } else {
            panic!("unhandled char {c}");
        }
    }
}

struct Parser<'a> {
    input: std::iter::Peekable<Lexer<'a>>,
    prog: Program,
    fun: FunId,
    block: BlockId,
    funs: HashMap<String, FunId>,
}

impl<'a> Parser<'a> {
    fn from_lexer(lexer: Lexer) -> Parser {
        let mut prog = Program::default();
        let (main_id, main_entry) = prog.new_fun();
        Parser { input: lexer.peekable(), prog: Program::default(), fun: main_id, block: main_entry, funs: HashMap::default() }
    }

    fn parse_program(&mut self) {
        while let Some(token) = self.input.next() {
            if token != Token::Def {
                panic!("Unexpected token {token:?}");
            }
            self.parse_fun();
        }
    }

    fn expect(&mut self, expected: Token) {
        match self.input.next() {
            Some(actual) if actual == expected => {},
            actual => panic!("Unexpected token {actual:?}"),
        }
    }

    fn parse_fun(&mut self) {
        let (name, (fun_id, block_id)) = match self.input.next() {
            Some(Token::Ident(name)) => (name.clone(), self.prog.new_fun_with_name(name)),
            token => panic!("Unexpected token {token:?}"),
        };
        self.fun = fun_id;
        self.block = block_id;
        self.expect(Token::LParen);
        let mut params: Vec<String> = vec![];
        loop {
            match self.input.peek () {
                Some(Token::RParen) => { break; }
                Some(Token::Ident(param)) => {
                    params.push(param.clone());
                    self.input.next();
                }
                token => panic!("Unexpected token {token:?}"),
            }
            match self.input.peek() {
                Some(Token::Comma) => { self.input.next(); }
                _ => { break; }
            }
        }
        self.expect(Token::RParen);
        let mut env: HashMap<String, Opnd> = HashMap::new();
        for (idx, param) in params.iter().enumerate() {
            let insn_id = self.prog.push_insn(block_id, Op::Param { idx });
            env.insert(param.clone(), Opnd::Insn(insn_id));
        }
        loop {
            match self.input.peek() {
                Some(Token::End) => { break; }
                Some(_) => { self.parse_statement(&mut env); }
                None => panic!("Unexpected EOF while parsing function"),
            }
        }
        if !self.prog.block_is_terminated(self.block) {
            self.prog.push_insn(self.block, Op::Return { val: Opnd::Const(Value::Nil) });
        }
        self.expect(Token::End);
        self.funs.insert(name, fun_id);
    }

    fn join_vars(&mut self, mut env: &mut HashMap<String, Opnd>,
                 left_block: BlockId, left_env: &HashMap<String, Opnd>,
                 right_block: BlockId, right_env: &HashMap<String, Opnd>) {
        let all_keys_set: HashSet<&String> = HashSet::from_iter(left_env.keys().chain(right_env.keys()));
        let mut all_keys: Vec<&String> = all_keys_set.into_iter().collect();
        all_keys.sort();  // Stable Phi ordering
        for key in all_keys {
            match (left_env.get(key), right_env.get(key)) {
                (Some(left), Some(right)) if left == right => {}
                (Some(left), Some(right)) => {
                    let phi = self.prog.push_insn(self.block, Op::Phi { ins: vec![
                        (left_block, *left),
                        (right_block, *right),
                    ] });
                    env.insert(key.clone(), Opnd::Insn(phi));
                }
                (Some(left), None) => {
                    let phi = self.prog.push_insn(self.block, Op::Phi { ins: vec![
                        (left_block, *left),
                        (right_block, Opnd::Const(Value::Nil)),
                    ] });
                    env.insert(key.clone(), Opnd::Insn(phi));
                }
                (None, Some(right)) => {
                    let phi = self.prog.push_insn(self.block, Op::Phi { ins: vec![
                        (left_block, Opnd::Const(Value::Nil)),
                        (right_block, *right),
                    ] });
                    env.insert(key.clone(), Opnd::Insn(phi));
                }
                (None, None) => panic!("Should not happen"),
            }
        }
    }

    fn parse_statement(&mut self, mut env: &mut HashMap<String, Opnd>) {
        match self.input.peek() {
            Some(Token::Return) => {
                self.input.next();
                let val = self.parse_expression(&mut env);
                self.prog.push_insn(self.block, Op::Return { val });
            }
            Some(Token::If) => {
                self.input.next();
                let val = self.parse_expression(&mut env);
                let mut then_env = env.clone();
                let if_block = self.block;
                let then_block = self.prog.new_block(self.fun);
                let join_block = self.prog.new_block(self.fun);
                self.block = then_block;
                loop {
                    match self.input.peek() {
                        Some(Token::End) => {
                            self.input.next();
                            self.prog.push_insn(if_block, Op::IfTrue { val, then_block, else_block: join_block });
                            self.prog.push_insn(self.block, Op::Jump { target: join_block });
                            self.block = join_block;
                            let if_env = env.clone();
                            self.join_vars(&mut env, if_block, &if_env, then_block, &then_env);
                            return;
                        }
                        Some(Token::Else) => {
                            self.input.next();
                            break;
                        }
                        _ => {
                            self.parse_statement(&mut then_env);
                        }
                    }
                }
                let mut else_env = env.clone();
                let else_block = self.prog.new_block(self.fun);
                self.prog.push_insn(self.block, Op::Jump { target: else_block });
                self.prog.push_insn(if_block, Op::IfTrue { val, then_block, else_block });
                self.block = else_block;
                loop {
                    match self.input.peek() {
                        Some(Token::End) => {
                            self.input.next();
                            self.prog.push_insn(self.block, Op::Jump { target: join_block });
                            self.block = join_block;
                            self.join_vars(&mut env, then_block, &then_env, else_block, &else_env);
                            return;
                        }
                        _ => {
                            self.parse_statement(&mut else_env);
                        }
                    }
                }
            }
            _ => { self.parse_expression(&mut env); }
        }
    }

    fn parse_expression(&mut self, mut env: &mut HashMap<String, Opnd>) -> Opnd{
        self.parse_(&mut env, 0)
    }

    fn prec(token: &Token) -> i8 {
        match token {
            Token::Plus => 1,
            Token::Minus => 1,
            // TODO(max): Unary negate?
            Token::Mul => 3,
            Token::Div => 3,
            _ => panic!("Don't know precedence of {token:?}"),
        }
    }

    fn assoc(token: &Token) -> Assoc {
        match token {
            Token::Plus => Assoc::Any,
            Token::Minus => Assoc::Left,
            // TODO(max): Unary negate?
            Token::Mul => Assoc::Any,
            Token::Div => Assoc::Left,
            _ => panic!("Don't know associativity of {token:?}"),
        }
    }

    fn paren(&mut self, mut env: &mut HashMap<String, Opnd>) -> Opnd {
        match self.input.peek() {
            None => panic!("Unexpected EOF"),
            Some(Token::Int(lit)) => {
                let lit = *lit;
                self.input.next();
                Opnd::Const(Value::Int(lit))
            }
            Some(Token::LParen) => {
                self.input.next();
                let result = self.parse_(&mut env, 0);
                self.expect(Token::RParen);
                result
            }
            token => panic!("Unexpected token {token:?}"),
        }
    }

    fn parse_(&mut self, mut env: &mut HashMap<String, Opnd>, prec: i8) -> Opnd {
        let mut lhs = match self.input.peek() {
            Some(Token::Ident(name)) => {
                let name = name.clone();
                self.input.next();
                match self.input.peek() {
                    // TODO(max): Something about precedence. Don't allow if > 0?
                    Some(Token::Equal) => {
                        // It's an assignment. The name isn't in the environment yet. Parse right
                        // to get the value so we can insert it.
                        self.input.next();
                        let rhs = self.parse_expression(&mut env);
                        if name == "true" || name == "false" || name == "nil" {
                            panic!("Can't assign to true/false/nil");
                        }
                        env.insert(name, rhs);
                        rhs
                    }
                    _ => {
                        // It's being used for its value.
                        if name == "true" { Opnd::Const(Value::Bool(true)) }
                        else if name == "false" { Opnd::Const(Value::Bool(false)) }
                        else if name == "nil" { Opnd::Const(Value::Nil) }
                        else {
                            self.funs.get(&name).and_then(|fun_id| Some(Opnd::Const(Value::Fun(*fun_id))))
                                .unwrap_or_else(|| *env.get(&name)
                                    .unwrap_or_else(|| panic!("Unbound name {name}")))
                        }
                    }
                }
            }
            _ => self.paren(&mut env),
        };
        loop {
            match self.input.peek() {
                Some(op @ (Token::Plus | Token::Minus | Token::Mul | Token::Div)) => {
                    let op = op.clone();
                    let op_prec = Self::prec(&op);
                    if op_prec < prec {
                        break;
                    }
                    self.input.next();
                    let next_prec = if Self::assoc(&op) == Assoc::Left { op_prec + 1 } else { op_prec };
                    let rhs = self.parse_(&mut env, next_prec);
                    lhs = Opnd::Insn(self.prog.push_insn(self.block, match op {
                        Token::Plus => Op::Add { v0: lhs, v1: rhs },
                        Token::Mul => Op::Mul { v0: lhs, v1: rhs },
                        _ => todo!(),
                    }));
                }
                Some(Token::LParen) => {
                    // Function call
                    self.input.next();
                    let mut args = vec![];
                    loop {
                        match self.input.peek() {
                            Some(Token::RParen) => { break; }
                            Some(_) => {
                                args.push(self.parse_(&mut env, 0));
                                if self.input.peek() == Some(&Token::Comma) {
                                    self.input.next();
                                    continue;
                                }
                                break;
                            }
                            _ => todo!(),
                        }
                    }
                    self.expect(Token::RParen);
                    lhs = match lhs {
                        Opnd::Const(Value::Fun(target)) =>
                            Opnd::Insn(self.prog.push_insn(self.block, Op::SendStatic { target, args })),
                        _ => panic!("Only static calls are supported (for now)"),
                    };
                }
                Some(Token::Dot) => {
                    // Method call
                    self.input.next();
                    let method = match self.input.next() {
                        Some(Token::Ident(method)) => method.clone(),
                        token => panic!("Unexpected token {token:?}"),
                    };
                    self.expect(Token::LParen);
                    let mut args = vec![];
                    loop {
                        match self.input.peek() {
                            Some(Token::RParen) => { break; }
                            Some(_) => {
                                args.push(self.parse_(&mut env, 0));
                                if self.input.peek() == Some(&Token::Comma) {
                                    self.input.next();
                                    continue;
                                }
                                break;
                            }
                            _ => todo!(),
                        }
                    }
                    self.expect(Token::RParen);
                    lhs = Opnd::Insn(self.prog.push_insn(self.block, Op::SendDynamic { method, self_val: lhs, args }));
                }
                Some(_) | None => { break; }
            }
        }
        lhs
    }
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
        assert_eq!(union(&Type::Const(Value::Int(3)), &Type::Const(Value::Bool(true))), Type::objects(&vec![INT_CLASS, TRUE_CLASS]));
    }

    #[test]
    fn test_type() {
        assert_eq!(union(&Type::Const(Value::Int(3)), &Type::Const(Value::Int(4))), Type::Int);
        assert_eq!(union(&Type::Const(Value::Int(3)), &Type::Int), Type::Int);

        assert_eq!(union(&Type::Const(Value::Bool(true)), &Type::Const(Value::Bool(false))), Type::Bool);
        assert_eq!(union(&Type::Const(Value::Bool(true)), &Type::Bool), Type::Bool);

        assert_eq!(union(&Type::Int, &Type::Bool), Type::objects(&vec![INT_CLASS, TRUE_CLASS, FALSE_CLASS]));
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
        let ret_id = prog.push_insn(block_id, Op::Return { val: Opnd::Insn(add_id) });
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
        assert_eq!(result.type_of(phi_id), Type::objects(&vec![TRUE_CLASS, INT_CLASS]));
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
        assert_eq!(result.type_of(phi_id), Type::objects(&vec![INT_CLASS, TRUE_CLASS]));
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
        let return_id = prog.push_insn(target_entry, Op::Return { val: Opnd::Const(Value::Int(5)) });
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
        let param_id = prog.push_insn(target_entry, Op::Param { idx: 0 });
        let return_id = prog.push_insn(target_entry, Op::Return { val: Opnd::Insn(param_id) });
        let send_id = prog.push_insn(block_id, Op::SendStatic { target, args: vec![Opnd::Const(Value::Int(5))] });
        let result = sctp(&mut prog);
        assert_eq!(result.is_executable(target_entry), true);
        assert_eq!(result.type_of(param_id), Type::Const(Value::Int(5)));
    }

    #[test]
    fn test_two_sends_flow_to_param() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let (target, target_entry) = prog.new_fun();
        let param_id = prog.push_insn(target_entry, Op::Param { idx: 0 });
        let return_id = prog.push_insn(target_entry, Op::Return { val: Opnd::Insn(param_id) });
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
        let param0_id = prog.push_insn(target_entry, Op::Param { idx: 0 });
        let param1_id = prog.push_insn(target_entry, Op::Param { idx: 1 });
        let add_id = prog.push_insn(target_entry, Op::Add { v0: Opnd::Insn(param0_id), v1: Opnd::Insn(param1_id) });
        let return_id = prog.push_insn(target_entry, Op::Return { val: Opnd::Insn(add_id) });
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
        prog.push_insn(end_id, Op::Return { val: Opnd::Insn(n) });
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
        let n = prog.push_insn(fact_entry, Op::Param { idx: 0 });
        let lt = prog.push_insn(fact_entry, Op::LessThan { v0: Opnd::Insn(n), v1: TWO });
        let early_exit_id = prog.new_block(fact_id);
        let do_mul_id = prog.new_block(fact_id);
        prog.push_insn(fact_entry, Op::IfTrue { val: Opnd::Insn(lt), then_block: early_exit_id, else_block: do_mul_id });
        prog.push_insn(early_exit_id, Op::Return { val: ONE });
        let sub = prog.push_insn(do_mul_id, Op::Add { v0: Opnd::Insn(n), v1: Opnd::Const(Value::Int(-1)) });
        let rec = prog.push_insn(do_mul_id, Op::SendStatic { target: fact_id, args: vec![Opnd::Insn(sub)] });
        let mul = prog.push_insn(do_mul_id, Op::Mul { v0: Opnd::Insn(n), v1: Opnd::Insn(rec) });
        prog.push_insn(do_mul_id, Op::Return { val: Opnd::Insn(mul) });
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
        let n = prog.push_insn(fib_entry, Op::Param { idx: 0 });
        let lt = prog.push_insn(fib_entry, Op::LessThan { v0: Opnd::Insn(n), v1: TWO });
        let early_exit_id = prog.new_block(fib_id);
        let do_add_id = prog.new_block(fib_id);
        prog.push_insn(fib_entry, Op::IfTrue { val: Opnd::Insn(lt), then_block: early_exit_id, else_block: do_add_id });
        prog.push_insn(early_exit_id, Op::Return { val: Opnd::Insn(n) });
        let sub1 = prog.push_insn(do_add_id, Op::Add { v0: Opnd::Insn(n), v1: Opnd::Const(Value::Int(-1)) });
        let rec1 = prog.push_insn(do_add_id, Op::SendStatic { target: fib_id, args: vec![Opnd::Insn(sub1)] });
        let sub2 = prog.push_insn(do_add_id, Op::Add { v0: Opnd::Insn(n), v1: Opnd::Const(Value::Int(-2)) });
        let rec2 = prog.push_insn(do_add_id, Op::SendStatic { target: fib_id, args: vec![Opnd::Insn(sub2)] });
        let add = prog.push_insn(do_add_id, Op::Add { v0: Opnd::Insn(rec1), v1: Opnd::Insn(rec2) });
        prog.push_insn(do_add_id, Op::Return { val: Opnd::Insn(add) });
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
        let x = prog.push_insn(callee_entry_id, Op::Param { idx: 0 });
        prog.push_insn(callee_entry_id, Op::Jump { target: phi_block_id });

        let t = prog.push_insn(dead_block_id, Op::Add { v0: Opnd::Insn(x), v1: Opnd::Const(Value::Int(1)) });
        let t2 = prog.push_insn(dead_block_id, Op::IsNil { v: Opnd::Insn(t) });
        prog.push_insn(dead_block_id, Op::IfTrue { val: Opnd::Insn(t2), then_block: phi_block_id, else_block: dead_block_id });

        let phi = prog.push_insn(phi_block_id, Op::Phi { ins: vec![(callee_entry_id, Opnd::Const(Value::Int(0))), (dead_block_id, Opnd::Const(Value::Int(1337)))] });
        prog.push_insn(phi_block_id, Op::Return { val: Opnd::Insn(phi) });

        let result = sctp(&mut prog);
        assert_eq!(result.type_of(phi), Type::Const(Value::Int(0)));
        assert_eq!(result.type_of(outside_call), Type::Const(Value::Int(0)));
    }

    #[test]
    fn test_new() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let obj = prog.push_insn(block_id, Op::New { class: ClassId(3) });
        prog.push_insn(block_id, Op::Return { val: Opnd::Insn(obj) });
        let result = sctp(&mut prog);
        assert_eq!(result.type_of(obj), Type::object(ClassId(3)));
    }

    #[test]
    fn test_phi_new() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let obj3 = prog.push_insn(block_id, Op::New { class: ClassId(3) });
        let obj4 = prog.push_insn(block_id, Op::New { class: ClassId(4) });
        let phi = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::Insn(obj3)), (block_id, Opnd::Insn(obj4))] });
        prog.push_insn(block_id, Op::Return { val: Opnd::Insn(phi) });
        let result = sctp(&mut prog);
        assert_eq!(result.type_of(phi), Type::objects(&vec![ClassId(3), ClassId(4)]));

    }

    #[test]
    fn test_one_send_dynamic_flows_to_method_param() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let class_id = prog.new_class();
        let (method_id, method_entry_id) = prog.new_method(class_id, "foo".into());
        let param_id = prog.push_insn(method_entry_id, Op::Param { idx: 0 });
        let return_id = prog.push_insn(method_entry_id, Op::Return { val: Opnd::Insn(param_id) });
        let obj_id = prog.push_insn(block_id, Op::New { class: class_id });
        let send_id = prog.push_insn(block_id, Op::SendDynamic { method: "foo".into(), self_val: Opnd::Insn(obj_id), args: vec![Opnd::Const(Value::Int(5))] });
        let result = sctp(&mut prog);
        assert_eq!(result.is_executable(method_entry_id), true);
        assert_eq!(result.type_of(param_id), Type::Const(Value::Int(5)));
    }

    #[test]
    fn test_send_dynamic_flows_to_all_potential_methods() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let class0_id = prog.new_class();
        let (method0_id, method0_entry_id) = prog.new_method(class0_id, "foo".into());
        let return0_id = prog.push_insn(method0_entry_id, Op::Return { val: Opnd::Const(Value::Int(3)) });
        let class1_id = prog.new_class();
        let (method1_id, method1_entry_id) = prog.new_method(class1_id, "foo".into());
        let return1_id = prog.push_insn(method1_entry_id, Op::Return { val: Opnd::Const(Value::Int(4)) });
        let obj0 = prog.push_insn(block_id, Op::New { class: class0_id });
        let obj1 = prog.push_insn(block_id, Op::New { class: class1_id });
        let phi = prog.push_insn(block_id, Op::Phi { ins: vec![(block_id, Opnd::Insn(obj0)), (block_id, Opnd::Insn(obj1))] });
        let send_id = prog.push_insn(block_id, Op::SendDynamic { method: "foo".into(), self_val: Opnd::Insn(phi), args: vec![Opnd::Const(Value::Int(5))] });
        prog.push_insn(block_id, Op::Return { val: Opnd::Insn(phi) });
        let result = sctp(&mut prog);
        assert_eq!(result.type_of(phi), Type::objects(&vec![class0_id, class1_id]));
        assert_eq!(result.type_of(send_id), Type::Int);
    }

    #[test]
    fn test_analyze_ctor_empty() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let (class, (ctor_fun_id, ctor_block_id)) = prog.new_class_with_ctor("C".into());
        prog.push_insn(ctor_block_id, Op::SelfParam { class_id: class });
        prog.push_insn(ctor_block_id, Op::Return { val: Opnd::Const(Value::Int(3)) });
        let result = analyze_ctor(&prog, class);
        assert_eq!(result, SmallBitSet(0));
    }

    #[test]
    fn test_analyze_set_ivar() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let (class, (ctor_fun_id, ctor_block_id)) = prog.new_class_with_ctor("C".into());
        prog.push_ivar(class, "bar".into());
        prog.push_ivar(class, "foo".into());
        let self_id = prog.push_insn(ctor_block_id, Op::SelfParam { class_id: class });
        prog.push_insn(ctor_block_id, Op::SetIvar { name: "foo".into(), self_val: Opnd::Insn(self_id), val: Opnd::Const(Value::Int(4)) });
        prog.push_insn(ctor_block_id, Op::Return { val: Opnd::Const(Value::Int(3)) });
        let result = analyze_ctor(&prog, class);
        assert_eq!(result, SmallBitSet(0b10));
    }

    #[test]
    fn test_analyze_set_ivar_one_branch() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let (class, (ctor_fun_id, ctor_block_id)) = prog.new_class_with_ctor("C".into());
        prog.push_ivar(class, "foo".into());
        prog.push_ivar(class, "bar".into());
        let self_id = prog.push_insn(ctor_block_id, Op::SelfParam { class_id: class });
        let left = prog.new_block(ctor_fun_id);
        let right = prog.new_block(ctor_fun_id);
        prog.push_insn(ctor_block_id, Op::IfTrue { val: Opnd::Const(Value::Int(3)), then_block: left, else_block: right });
        prog.push_insn(left, Op::SetIvar { name: "foo".into(), self_val: Opnd::Insn(self_id), val: Opnd::Const(Value::Int(4)) });
        prog.push_insn(left, Op::SetIvar { name: "bar".into(), self_val: Opnd::Insn(self_id), val: Opnd::Const(Value::Int(4)) });
        let join = prog.new_block(ctor_fun_id);
        prog.push_insn(left, Op::Jump { target: join });
        prog.push_insn(right, Op::SetIvar { name: "foo".into(), self_val: Opnd::Insn(self_id), val: Opnd::Const(Value::Int(4)) });
        prog.push_insn(right, Op::Jump { target: join });
        prog.push_insn(join, Op::Return { val: Opnd::Const(Value::Int(3)) });
        let result = analyze_ctor(&prog, class);
        assert_eq!(result, SmallBitSet(0b01));
    }

    #[test]
    fn test_ivar_types() {
        let (mut prog, fun_id, block_id) = prog_with_empty_fun();
        let (class, (ctor_fun_id, ctor_block_id)) = prog.new_class_with_ctor("C".into());
        prog.push_ivar(class, "foo".into());
        prog.push_ivar(class, "bar".into());
        let self_id = prog.push_insn(ctor_block_id, Op::SelfParam { class_id: class });
        prog.push_insn(ctor_block_id, Op::SetIvar { name: "foo".into(), self_val: Opnd::Insn(self_id), val: Opnd::Const(Value::Int(4)) });
        prog.push_insn(ctor_block_id, Op::SetIvar { name: "bar".into(), self_val: Opnd::Insn(self_id), val: Opnd::Const(Value::Bool(true)) });
        prog.push_insn(ctor_block_id, Op::Return { val: Opnd::Const(Value::Int(3)) });

        let obj = prog.create_instance(block_id, class);
        let getivar_foo_id = prog.push_insn(block_id, Op::GetIvar { name: "foo".into(), self_val: Opnd::Insn(obj) });
        let getivar_bar_id = prog.push_insn(block_id, Op::GetIvar { name: "bar".into(), self_val: Opnd::Insn(obj) });
        prog.push_insn(block_id, Op::Return { val: Opnd::Insn(getivar_foo_id) });

        let result = sctp(&mut prog);
        assert_eq!(result.type_of(getivar_foo_id), Type::Const(Value::Int(4)));
        assert_eq!(result.type_of(getivar_bar_id), Type::Const(Value::Bool(true)));
    }
}

#[cfg(test)]
mod parser_tests {
    use super::*;

    #[test]
    fn test_empty_returns_eof() {
        let mut lexer = Lexer::new("");
        assert_eq!(lexer.next(), None);
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_def() {
        let mut lexer = Lexer::new("   def");
        assert_eq!(lexer.next(), Some(Token::Def));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_end() {
        let mut lexer = Lexer::new("   end");
        assert_eq!(lexer.next(), Some(Token::End));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_return() {
        let mut lexer = Lexer::new("   return");
        assert_eq!(lexer.next(), Some(Token::Return));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_if() {
        let mut lexer = Lexer::new("   if");
        assert_eq!(lexer.next(), Some(Token::If));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_else() {
        let mut lexer = Lexer::new("   else");
        assert_eq!(lexer.next(), Some(Token::Else));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_ident() {
        let mut lexer = Lexer::new("   abc");
        assert_eq!(lexer.next(), Some(Token::Ident("abc".into())));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_parens() {
        let mut lexer = Lexer::new("   ()");
        assert_eq!(lexer.next(), Some(Token::LParen));
        assert_eq!(lexer.next(), Some(Token::RParen));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_comma() {
        let mut lexer = Lexer::new("   ,");
        assert_eq!(lexer.next(), Some(Token::Comma));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_plus() {
        let mut lexer = Lexer::new("   +");
        assert_eq!(lexer.next(), Some(Token::Plus));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_mul() {
        let mut lexer = Lexer::new("   *");
        assert_eq!(lexer.next(), Some(Token::Mul));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_equal() {
        let mut lexer = Lexer::new("   =");
        assert_eq!(lexer.next(), Some(Token::Equal));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_dot() {
        let mut lexer = Lexer::new("   .");
        assert_eq!(lexer.next(), Some(Token::Dot));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_digit() {
        let mut lexer = Lexer::new("   1");
        assert_eq!(lexer.next(), Some(Token::Int(1)));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_int() {
        let mut lexer = Lexer::new("   123");
        assert_eq!(lexer.next(), Some(Token::Int(123)));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_int_add() {
        let mut lexer = Lexer::new("   123+456");
        assert_eq!(lexer.next(), Some(Token::Int(123)));
        assert_eq!(lexer.next(), Some(Token::Plus));
        assert_eq!(lexer.next(), Some(Token::Int(456)));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_function() {
        let mut lexer = Lexer::new("def foo(a, b) a+b end");
        use Token::*;
        assert_eq!(lexer.collect::<Vec<_>>(), vec![
            Def, Ident("foo".into()), LParen, Ident("a".into()), Comma, Ident("b".into()), RParen,
            Ident("a".into()), Plus, Ident("b".into()),
            End
        ]);
    }

    #[test]
    fn test_parse_empty_function() {
        let mut lexer = Lexer::new("def foo() end");
        use Token::*;
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        assert_eq!(parser.prog.funs.len(), 1);
        assert_eq!(parser.prog.funs[0].name, "foo");
    }

    #[test]
    fn test_parse_true() {
        let mut lexer = Lexer::new("def foo() return true end");
        use Token::*;
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        assert_eq!(parser.prog.funs.len(), 1);
        assert_eq!(parser.prog.funs[0].name, "foo");
        assert_eq!(parser.prog.insns[0].op, Op::Return { val: Opnd::Const(Value::Bool(true)) });
    }

    #[test]
    fn test_parse_false() {
        let mut lexer = Lexer::new("def foo() return false end");
        use Token::*;
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        assert_eq!(parser.prog.funs.len(), 1);
        assert_eq!(parser.prog.funs[0].name, "foo");
        assert_eq!(parser.prog.insns[0].op, Op::Return { val: Opnd::Const(Value::Bool(false)) });
    }

    #[test]
    fn test_parse_nil() {
        let mut lexer = Lexer::new("def foo() return nil end");
        use Token::*;
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        assert_eq!(parser.prog.funs.len(), 1);
        assert_eq!(parser.prog.funs[0].name, "foo");
        assert_eq!(parser.prog.insns[0].op, Op::Return { val: Opnd::Const(Value::Nil) });
    }

    #[test]
    fn test_parse_function_with_params() {
        let mut lexer = Lexer::new("def foo(a, b) end");
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        let prog = parser.prog;
        assert_eq!(prog.funs.len(), 1);
        assert_eq!(prog.funs[0].name, "foo");
        assert_eq!(prog.insns[0].op, Op::Param { idx: 0 });
        assert_eq!(prog.insns[1].op, Op::Param { idx: 1 });
        assert_eq!(prog.insns[2].op, Op::Return { val: Opnd::Const(Value::Nil) });
    }

    #[test]
    fn test_parse_two_functions() {
        let mut lexer = Lexer::new("
def foo(a, b) end
def bar(c, d) end
");
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        let prog = parser.prog;
        assert_eq!(prog.funs.len(), 2);
        assert_eq!(prog.funs[0].name, "foo");
        assert_eq!(prog.insns[0].op, Op::Param { idx: 0 });
        assert_eq!(prog.insns[1].op, Op::Param { idx: 1 });
        assert_eq!(prog.insns[2].op, Op::Return { val: Opnd::Const(Value::Nil) });

        assert_eq!(prog.funs[1].name, "bar");
        assert_eq!(prog.insns[3].op, Op::Param { idx: 0 });
        assert_eq!(prog.insns[4].op, Op::Param { idx: 1 });
        assert_eq!(prog.insns[5].op, Op::Return { val: Opnd::Const(Value::Nil) });
    }

    #[test]
    fn test_parse_function_const_add() {
        let mut lexer = Lexer::new("def add() 1+2 end");
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        let prog = parser.prog;
        assert_eq!(prog.funs.len(), 1);
        assert_eq!(prog.funs[0].name, "add");
        assert_eq!(prog.insns[0].op, Op::Add { v0: Opnd::Const(Value::Int(1)), v1: Opnd::Const(Value::Int(2)) });
        assert_eq!(prog.insns[1].op, Op::Return { val: Opnd::Const(Value::Nil) });
    }

    #[test]
    fn test_parse_function_add() {
        let mut lexer = Lexer::new("def add(a, b) a+b end");
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        let prog = parser.prog;
        assert_eq!(prog.funs.len(), 1);
        assert_eq!(prog.funs[0].name, "add");
        assert_eq!(prog.insns[0].op, Op::Param { idx: 0 });
        assert_eq!(prog.insns[1].op, Op::Param { idx: 1 });
        assert_eq!(prog.insns[2].op, Op::Add { v0: Opnd::Insn(InsnId(0)), v1: Opnd::Insn(InsnId(1)) });
        assert_eq!(prog.insns[3].op, Op::Return { val: Opnd::Const(Value::Nil) });
    }

    #[test]
    fn test_parse_function_add_mul() {
        let mut lexer = Lexer::new("def add(a, b, c) a+b*c end");
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        let prog = parser.prog;
        assert_eq!(prog.funs.len(), 1);
        assert_eq!(prog.funs[0].name, "add");
        assert_eq!(prog.insns[0].op, Op::Param { idx: 0 });
        assert_eq!(prog.insns[1].op, Op::Param { idx: 1 });
        assert_eq!(prog.insns[2].op, Op::Param { idx: 2 });
        assert_eq!(prog.insns[3].op, Op::Mul { v0: Opnd::Insn(InsnId(1)), v1: Opnd::Insn(InsnId(2)) });
        assert_eq!(prog.insns[4].op, Op::Add { v0: Opnd::Insn(InsnId(0)), v1: Opnd::Insn(InsnId(3)) });
        assert_eq!(prog.insns[5].op, Op::Return { val: Opnd::Const(Value::Nil) });
    }

    #[test]
    fn test_parse_function_mul_add() {
        let mut lexer = Lexer::new("def add(a, b, c) a*b+c end");
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        let prog = parser.prog;
        assert_eq!(prog.funs.len(), 1);
        assert_eq!(prog.funs[0].name, "add");
        assert_eq!(prog.insns[0].op, Op::Param { idx: 0 });
        assert_eq!(prog.insns[1].op, Op::Param { idx: 1 });
        assert_eq!(prog.insns[2].op, Op::Param { idx: 2 });
        assert_eq!(prog.insns[3].op, Op::Mul { v0: Opnd::Insn(InsnId(0)), v1: Opnd::Insn(InsnId(1)) });
        assert_eq!(prog.insns[4].op, Op::Add { v0: Opnd::Insn(InsnId(3)), v1: Opnd::Insn(InsnId(2)) });
        assert_eq!(prog.insns[5].op, Op::Return { val: Opnd::Const(Value::Nil) });
    }

    #[test]
    fn test_parse_function_explicit_return() {
        let mut lexer = Lexer::new("def foo() return 1 end");
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        let prog = parser.prog;
        assert_eq!(prog.funs.len(), 1);
        assert_eq!(prog.funs[0].name, "foo");
        assert_eq!(prog.insns[0].op, Op::Return { val: Opnd::Const(Value::Int(1)) });
    }

    #[test]
    fn test_parse_function_implicit_return_nil() {
        // TODO(max): Maybe consider making blocks return value of last expression
        let mut lexer = Lexer::new("def foo() 1 end");
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        let prog = parser.prog;
        assert_eq!(prog.funs.len(), 1);
        assert_eq!(prog.funs[0].name, "foo");
        assert_eq!(prog.insns[0].op, Op::Return { val: Opnd::Const(Value::Nil) });
    }

    #[test]
    fn test_parse_function_assign() {
        let mut lexer = Lexer::new("
def foo()
  a = 123
  return a+1
end");
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        let prog = parser.prog;
        assert_eq!(prog.funs.len(), 1);
        assert_eq!(prog.funs[0].name, "foo");
        assert_eq!(prog.insns[0].op, Op::Add { v0: Opnd::Const(Value::Int(123)), v1: Opnd::Const(Value::Int(1)) });
        assert_eq!(prog.insns[1].op, Op::Return { val: Opnd::Insn(InsnId(0)) });
    }

    fn assert_block_equals(prog: &Program, block: BlockId, ops: Vec<Op>) {
        assert_eq!(prog.blocks[block.0].insns.len(), ops.len(), "Block length mismatch");
        for (idx, insn_id) in prog.blocks[block.0].insns.iter().enumerate() {
            assert_eq!(prog.insns[insn_id.0].op, ops[idx], "Argument mismatch");
        }
    }

    #[test]
    fn test_parse_empty_if() {
        let mut lexer = Lexer::new("
def foo()
  if 1
  end
end");
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        let prog = parser.prog;
        assert_eq!(prog.funs.len(), 1);
        assert_eq!(prog.funs[0].name, "foo");
        assert_block_equals(&prog, prog.funs[0].entry_block, vec![
            Op::IfTrue { val: Opnd::Const(Value::Int(1)), then_block: BlockId(1), else_block: BlockId(2) },
        ]);
        assert_block_equals(&prog, BlockId(1), vec![
            Op::Jump { target: BlockId(2) },
        ]);
        assert_block_equals(&prog, BlockId(2), vec![
            Op::Return { val: Opnd::Const(Value::Nil) },
        ]);
    }

    #[test]
    fn test_parse_empty_if_else() {
        let mut lexer = Lexer::new("
def foo()
  if 1
  else
  end
end");
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        let prog = parser.prog;
        assert_eq!(prog.funs.len(), 1);
        assert_eq!(prog.funs[0].name, "foo");
        assert_block_equals(&prog, prog.funs[0].entry_block, vec![
            Op::IfTrue { val: Opnd::Const(Value::Int(1)), then_block: BlockId(1), else_block: BlockId(3) },
        ]);
        assert_block_equals(&prog, BlockId(1), vec![
            Op::Jump { target: BlockId(3) },
        ]);
        assert_block_equals(&prog, BlockId(2), vec![
            Op::Return { val: Opnd::Const(Value::Nil) },
        ]);
        assert_block_equals(&prog, BlockId(3), vec![
            Op::Jump { target: BlockId(2) },
        ]);
    }

    #[test]
    fn test_if_phi() {
        // a is defined before and modified in both branches
        // b is defiend before and modified only in then
        // c is defined before and modified only ine lse
        // d is defined in then and in else
        // e is defined before and not modified
        // f is defined only in then
        // g is defined only in else
        let mut lexer = Lexer::new("
def foo()
  a = 1
  b = 4
  c = 6
  e = 10
  if 1
    a = 2
    b = 5
    d = 8
    f = 11
  else
    a = 3
    c = 7
    d = 9
    g = 12
  end
  return a+b+c+d+e+f+g
end");
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        let prog = parser.prog;
        assert_eq!(prog.funs.len(), 1);
        assert_eq!(prog.funs[0].name, "foo");
        assert_block_equals(&prog, BlockId(2), vec![
            Op::Phi { ins: vec![
                (BlockId(1), Opnd::Const(Value::Int(2))),
                (BlockId(3), Opnd::Const(Value::Int(3))),
            ] },
            Op::Phi { ins: vec![
                (BlockId(1), Opnd::Const(Value::Int(5))),
                (BlockId(3), Opnd::Const(Value::Int(4))),
            ] },
            Op::Phi { ins: vec![
                (BlockId(1), Opnd::Const(Value::Int(6))),
                (BlockId(3), Opnd::Const(Value::Int(7))),
            ] },
            Op::Phi { ins: vec![
                (BlockId(1), Opnd::Const(Value::Int(8))),
                (BlockId(3), Opnd::Const(Value::Int(9))),
            ] },
            Op::Phi { ins: vec![
                (BlockId(1), Opnd::Const(Value::Int(11))),
                (BlockId(3), Opnd::Const(Value::Nil)),
            ] },
            Op::Phi { ins: vec![
                (BlockId(1), Opnd::Const(Value::Nil)),
                (BlockId(3), Opnd::Const(Value::Int(12))),
            ] },
            Op::Add { v0: Opnd::Insn(InsnId(7)), v1: Opnd::Insn(InsnId(8)) },
            Op::Add { v0: Opnd::Const(Value::Int(10)), v1: Opnd::Insn(InsnId(9)) },
            Op::Add { v0: Opnd::Insn(InsnId(6)), v1: Opnd::Insn(InsnId(10)) },
            Op::Add { v0: Opnd::Insn(InsnId(5)), v1: Opnd::Insn(InsnId(11)) },
            Op::Add { v0: Opnd::Insn(InsnId(4)), v1: Opnd::Insn(InsnId(12)) },
            Op::Add { v0: Opnd::Insn(InsnId(3)), v1: Opnd::Insn(InsnId(13)) },
            Op::Return { val: Opnd::Insn(InsnId(14)) },
        ]);
    }

    #[test]
    fn test_parse_call_no_args() {
        let mut lexer = Lexer::new("
def bar()
end
def foo()
  bar()
end");
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        let prog = parser.prog;
        assert_eq!(prog.funs.len(), 2);
        assert_eq!(prog.funs[0].name, "bar");
        assert_eq!(prog.funs[1].name, "foo");
        assert_block_equals(&prog, prog.funs[0].entry_block, vec![
            Op::Return { val: Opnd::Const(Value::Nil) },
        ]);
        assert_block_equals(&prog, prog.funs[1].entry_block, vec![
            Op::SendStatic { target: FunId(0), args: vec![] },
            Op::Return { val: Opnd::Const(Value::Nil) },
        ]);
    }

    #[test]
    fn test_parse_call_one_arg() {
        let mut lexer = Lexer::new("
def bar()
end
def foo()
  bar(1)
end");
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        let prog = parser.prog;
        assert_eq!(prog.funs.len(), 2);
        assert_eq!(prog.funs[0].name, "bar");
        assert_eq!(prog.funs[1].name, "foo");
        assert_block_equals(&prog, prog.funs[0].entry_block, vec![
            Op::Return { val: Opnd::Const(Value::Nil) },
        ]);
        assert_block_equals(&prog, prog.funs[1].entry_block, vec![
            Op::SendStatic { target: FunId(0), args: vec![Opnd::Const(Value::Int(1))] },
            Op::Return { val: Opnd::Const(Value::Nil) },
        ]);
    }

    #[test]
    fn test_parse_call_multiple_args() {
        let mut lexer = Lexer::new("
def bar()
end
def foo()
  bar(1, 2, 3)
end");
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        let prog = parser.prog;
        assert_eq!(prog.funs.len(), 2);
        assert_eq!(prog.funs[0].name, "bar");
        assert_eq!(prog.funs[1].name, "foo");
        assert_block_equals(&prog, prog.funs[0].entry_block, vec![
            Op::Return { val: Opnd::Const(Value::Nil) },
        ]);
        assert_block_equals(&prog, prog.funs[1].entry_block, vec![
            Op::SendStatic { target: FunId(0), args: vec![
                Opnd::Const(Value::Int(1)),
                Opnd::Const(Value::Int(2)),
                Opnd::Const(Value::Int(3))
            ] },
            Op::Return { val: Opnd::Const(Value::Nil) },
        ]);
    }

    #[test]
    fn test_parse_call_method_no_args() {
        let mut lexer = Lexer::new("
def foo(o)
  o.bar()
end");
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        let prog = parser.prog;
        assert_eq!(prog.funs.len(), 1);
        assert_eq!(prog.funs[0].name, "foo");
        assert_block_equals(&prog, prog.funs[0].entry_block, vec![
            Op::Param { idx: 0 },
            Op::SendDynamic { method: "bar".into(), self_val: Opnd::Insn(InsnId(0)), args: vec![] },
            Op::Return { val: Opnd::Const(Value::Nil) },
        ]);
    }

    #[test]
    fn test_parse_call_method() {
        let mut lexer = Lexer::new("
def foo(o)
  o.bar(1, 2, 3)
end");
        let mut parser = Parser::from_lexer(lexer);
        parser.parse_program();
        let prog = parser.prog;
        assert_eq!(prog.funs.len(), 1);
        assert_eq!(prog.funs[0].name, "foo");
        assert_block_equals(&prog, prog.funs[0].entry_block, vec![
            Op::Param { idx: 0 },
            Op::SendDynamic { method: "bar".into(), self_val: Opnd::Insn(InsnId(0)), args: vec![
                Opnd::Const(Value::Int(1)),
                Opnd::Const(Value::Int(2)),
                Opnd::Const(Value::Int(3))
            ] },
            Op::Return { val: Opnd::Const(Value::Nil) },
        ]);
    }
}
