#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use std::collections::{HashMap, HashSet};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ClassDesc {
    name: String,

    // List of fields
    fields: Vec<String>,

    // List of methods
    methods: HashMap<String, FunId>,

    // Constructor method
    ctor: FunId,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct ClassId(usize);

// Function id
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct FunId(usize);

// The type an IR Insn can have.
#[derive(Eq, PartialEq, Debug, Clone)]
pub enum Type {
    Bottom, // Empty; no values possible; dead code
    Const(Value),
    Exact(ClassId),
    // TODO(max): Support limited (shallow) type unions because things will often be Integer|nil, etc
    // Either(Type, Type),
    Union(HashSet<ClassId>),
    // This would also help with TrueClass|FalseClass, since there's no bool type in Ruby.
    // No inheritance, otherwise we would also need an Inexact
    Top, // Unknown; could be anything
}

impl Type {
    pub fn empty() -> Type {
        Type::Bottom
    }

    pub fn from_value(value: &Value) -> Type {
        match value {
            Value::Nil => Type::Exact(NIL_TYPE),
            Value::Int(..) => Type::Exact(INT_TYPE),
            Value::Str(..) => Type::Exact(STR_TYPE),
            _ => todo!(),
        }
    }

    pub fn union(self: &Self, other: &Type) -> Type {
        use Type::*;
        match (self, other) {
            (Bottom, _) => other.clone(),
            (Top, _) => Top,
            (_, _) if self == other => self.clone(),
            (Const(l), Const(r)) => Type::from_value(l).union(&Type::from_value(r)),
            (Exact(left_class), Exact(right_class)) if left_class == right_class => self.clone(),
            (Exact(left_class), Exact(right_class)) => {
                Union(HashSet::from([*left_class, *right_class]))
            }
            (Union(set), Exact(new)) | (Exact(new), Union(set)) => {
                let mut result = set.clone();
                result.insert(*new);
                Union(result)
            }
            (_, _) => Top,
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Bottom => write!(f, "Bottom"),
            Type::Const(v) => write!(f, "Const[{v}]"),
            Type::Exact(class_id) => write!(f, "Class@{}", class_id.0),
            Type::Union(class_ids) =>
            // TODO(max): Assert size >= 2
            {
                write!(
                    f,
                    "{}",
                    class_ids
                        .into_iter()
                        .map(|id| format!("Class@{}", id.0))
                        .collect::<Vec<_>>()
                        .join("|")
                )
            }
            Type::Top => write!(f, "Top"),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Value {
    Nil,
    Int(i64),
    Str(String),
    Symbol(String),
    Fun(FunId),
    Class(ClassId),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Nil => write!(f, "Nil"),
            Value::Int(val) => write!(f, "{val}"),
            Value::Str(val) => write!(f, "{val:?}"),
            Value::Symbol(val) => write!(f, ":{val}"),
            Value::Class(id) => write!(f, "Class:{}", id.0),
            Value::Fun(id) => write!(f, "Fun:{}", id.0),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InsnId(usize);

impl std::fmt::Display for InsnId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Opnd {
    Param(usize),

    // Constant
    Const(Value),

    // Output of a previous insn in a block
    // that dominates this one
    InsnOut(InsnId),
}

impl std::fmt::Display for Opnd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Opnd::Const(val) => write!(f, "Const[{val}]"),
            Opnd::InsnOut(insn_id) => write!(f, "{insn_id}"),
            Opnd::Param(idx) => write!(f, "arg{idx}"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct BlockId(usize);

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

#[derive(Debug)]
pub struct JumpEdge {
    target: BlockId,
    opnds: Vec<Opnd>,
}

impl std::fmt::Display for JumpEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(", self.target)?;
        let mut sep = "";
        for opnd in &self.opnds {
            write!(f, "{sep}{opnd}")?;
            sep = ", ";
        }
        write!(f, ")")
    }
}

#[derive(Debug)]
pub enum Insn {
    // Send may need to branch to a return block (call continuation)
    // Send has an output
    // How do we handle phi nodes, branches following calls?
    // May be simpler if followed by a branch
    Send {
        receiver: Opnd,
        name: Opnd,
        args: Vec<Opnd>,
    },

    Return(Opnd),
    NewInstance(Opnd, Vec<Opnd>),
    IvarSet(Opnd, Opnd, Opnd),
    IvarGet(Opnd, Opnd),
    IsInt(Opnd),
    Add(Opnd, Opnd),
    Lt(Opnd, Opnd),

    // ?
    //RefineType(Opnd, Type)

    // Maxime says: for branches we're going to need
    // to supply block argumens for each target
    // we may also want to make IfTrue a one-sided branch for simplicity?
    // do we care about having only one final branch at the end of blocks?
    IfTrue(Opnd, JumpEdge, JumpEdge),
    Jump(JumpEdge),
}

impl Insn {
    pub fn is_terminator(&self) -> bool {
        use Insn::*;
        match self {
            Return(..) | IfTrue(..) | Jump(..) => true,
            _ => false,
        }
    }

    pub fn has_output(&self) -> bool {
        !self.is_terminator()
    }
}

fn fmt_args(f: &mut std::fmt::Formatter<'_>, args: &Vec<Opnd>) -> std::fmt::Result {
    for arg in args {
        write!(f, ", {arg}")?;
    }
    Ok(())
}

impl std::fmt::Display for Insn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Insn::Send {
                receiver,
                name,
                args,
            } => {
                write!(f, "Send {receiver}, {name}")?;
                fmt_args(f, args)
            }
            Insn::Return(opnd) => write!(f, "Return {opnd}"),
            Insn::NewInstance(cls, args) => {
                write!(f, "NewInstance {cls}")?;
                fmt_args(f, args)
            }
            Insn::IvarSet(receiver, attr, value) => {
                write!(f, "IvarSet {receiver}, {attr}, {value}")
            }
            Insn::IvarGet(receiver, attr) => write!(f, "IvarGet {receiver}, {attr}"),
            Insn::IsInt(opnd) => write!(f, "IsInt {opnd}"),
            Insn::Add(left, right) => write!(f, "Add {left}, {right}"),
            Insn::Lt(left, right) => write!(f, "Lt {left}, {right}"),
            Insn::IfTrue(cond, conseq, alt) => write!(f, "IfTrue {cond} then {conseq} else {alt}"),
            Insn::Jump(edge) => write!(f, "Jump {edge}"),
        }
    }
}

#[derive(Debug)]
pub struct Block {
    param_types: Vec<Type>,
    insns: Vec<InsnId>,
}

impl Block {
    pub fn empty() -> Block {
        Block {
            param_types: vec![],
            insns: vec![],
        }
    }

    pub fn with_params(num_params: usize) -> Block {
        Block {
            param_types: vec![Type::Bottom; num_params],
            insns: vec![],
        }
    }

    fn num_params(&self) -> usize {
        self.param_types.len()
    }
}

#[derive(Debug)]
pub struct ManagedFunction {
    entrypoint: BlockId,

    // Permanent home of every instruction; grows without bound
    insns: Vec<Insn>,

    // Type of each insn; index by InsnId just like insns
    insn_types: Vec<Type>,

    // Permanent home of every block; grows without bound
    blocks: Vec<Block>,
    // We may need to keep track of callers/successors of instructions
    // to implement an SCCP-like algorithm?
    // We need to be able to "push" type information forward when a type
    // changes

    // List of caller instructions
    //callers
}

impl ManagedFunction {
    pub fn new() -> ManagedFunction {
        let entry = Block::empty();
        ManagedFunction {
            entrypoint: BlockId(0),
            insns: vec![],
            insn_types: vec![],
            blocks: vec![entry],
        }
    }

    pub fn add_insn(&mut self, insn: Insn) -> InsnId {
        let result = InsnId(self.insns.len());
        self.insns.push(insn);
        self.insn_types.push(Type::Bottom);
        result
    }

    pub fn insn_at(&self, insn_id: InsnId) -> &Insn {
        &self.insns[insn_id.0]
    }

    pub fn alloc_block(&mut self) -> BlockId {
        let result = BlockId(self.blocks.len());
        self.blocks.push(Block::empty());
        result
    }

    pub fn alloc_block_with_params(&mut self, num_params: usize) -> BlockId {
        let result = BlockId(self.blocks.len());
        self.blocks.push(Block::with_params(num_params));
        result
    }

    pub fn push(&mut self, block: BlockId, insn: Insn) -> InsnId {
        let result = self.add_insn(insn);
        self.blocks[block.0].insns.push(result);
        result
    }

    fn type_of(&self, block_id: BlockId, opnd: &Opnd) -> Type {
        match opnd {
            Opnd::Const(v) => Type::Const(v.clone()),
            Opnd::InsnOut(id) => self.insn_types[id.0].clone(),
            Opnd::Param(idx) => self.blocks[block_id.0].param_types[*idx].clone(),
        }
    }

    fn reflow_insn(&self, block_id: BlockId, insn: &Insn) -> Type {
        use Opnd::*;
        match insn {
            Insn::Add(l, r) => match (self.type_of(block_id, l), self.type_of(block_id, r)) {
                (Type::Const(Value::Int(lv)), Type::Const(Value::Int(rv))) => {
                    Type::Const(Value::Int(lv + rv))
                }
                (Type::Exact(INT_TYPE), Type::Exact(INT_TYPE)) => Type::Exact(INT_TYPE),
                _ => Type::Top,
            },
            Insn::Lt(l, r) => match (self.type_of(block_id, l), self.type_of(block_id, r)) {
                (Type::Const(Value::Int(lv)), Type::Const(Value::Int(rv))) if lv < rv => {
                    Type::Exact(TRUE_TYPE)
                }
                (Type::Const(Value::Int(lv)), Type::Const(Value::Int(rv))) => {
                    Type::Exact(FALSE_TYPE)
                }
                (Type::Exact(INT_TYPE), Type::Exact(INT_TYPE)) => {
                    Type::Union(HashSet::from([TRUE_TYPE, FALSE_TYPE]))
                }
                _ => Type::Top,
            },
            _ => Type::Top,
        }
    }

    fn terminator_of(&self, block: BlockId) -> &Insn {
        let insn_id = self.blocks[block.0].insns.last().unwrap();
        &self.insns[insn_id.0]
    }

    pub fn rpo(&self) -> Vec<BlockId> {
        self.rpo_from(self.entrypoint)
    }

    fn rpo_from(&self, block: BlockId) -> Vec<BlockId> {
        let mut po_traversal = self.po_from(block);
        po_traversal.reverse();
        po_traversal
    }

    fn po_from(&self, block: BlockId) -> Vec<BlockId> {
        let mut result = vec![];
        let mut visited = HashSet::new();
        self.po_traverse_from(block, &mut result, &mut visited);
        result
    }

    fn po_traverse_from(
        &self,
        block: BlockId,
        result: &mut Vec<BlockId>,
        visited: &mut HashSet<BlockId>,
    ) {
        visited.insert(block);
        match self.terminator_of(block) {
            Insn::Return(_) => (),
            Insn::IfTrue(_, conseq, alt) => {
                if !visited.contains(&conseq.target) {
                    self.po_traverse_from(conseq.target, result, visited);
                }
                if !visited.contains(&alt.target) {
                    self.po_traverse_from(alt.target, result, visited);
                }
            }
            Insn::Jump(edge) => {
                if !visited.contains(&edge.target) {
                    self.po_traverse_from(edge.target, result, visited);
                }
            }
            insn => {
                panic!("Invalid terminator {insn}")
            }
        }
        result.push(block);
    }

    fn union_params(left: &Vec<Type>, right: Vec<Type>) -> Vec<Type> {
        left.iter()
            .zip(right.iter())
            .map(|(left_ty, right_ty)| left_ty.union(right_ty))
            .collect()
    }

    fn incoming_edges(&self, dst: BlockId) -> Vec<(BlockId, &JumpEdge)> {
        let mut result = vec![];
        for (idx, block) in self.blocks.iter().enumerate() {
            let block_id = BlockId(idx);
            match self.insn_at(*block.insns.last().unwrap()) {
                Insn::Jump(edge) => {
                    if edge.target == dst {
                        result.push((block_id, edge))
                    }
                }
                Insn::IfTrue(_, conseq, alt) => {
                    if conseq.target == dst {
                        result.push((block_id, conseq));
                    }
                    if alt.target == dst {
                        result.push((block_id, alt));
                    }
                }
                Insn::Return(_) => {}
                _ => todo!(),
            }
        }
        result
    }

    fn edge_types(&self, block_id: BlockId, edge: &JumpEdge) -> Vec<Type> {
        edge.opnds
            .iter()
            .map(|opnd| self.type_of(block_id, opnd))
            .collect()
    }

    pub fn reflow_types(&mut self) {
        // Reset all instruction types
        for ty in &mut self.insn_types {
            *ty = Type::Bottom;
        }
        // For each block in reverse post-order
        for block_id in self.rpo() {
            // Flow all incoming arguments to the block parameter types
            let mut param_types = vec![Type::Bottom; self.blocks[block_id.0].num_params()];
            for (from_id, edge) in self.incoming_edges(block_id) {
                param_types = Self::union_params(&param_types, self.edge_types(from_id, edge));
            }
            // Flow types through block's instructions
            self.blocks[block_id.0].param_types = param_types;
            for insn_id in &self.blocks[block_id.0].insns {
                self.insn_types[insn_id.0] = self.reflow_insn(block_id, self.insn_at(*insn_id));
            }
        }
    }
}

#[derive(Debug)]
pub struct NativeFunction(String);

impl std::fmt::Display for NativeFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug)]
pub enum Function {
    Managed(ManagedFunction),
    Native(NativeFunction),
}

impl std::fmt::Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Function::Managed(fun) => write!(f, "{}", fun),
            Function::Native(fun) => write!(f, "Native:{}", fun),
        }
    }
}

struct DisplayBlock<'a> {
    function: &'a ManagedFunction,
    block: &'a Block,
    indent: usize,
}

impl<'a> std::fmt::Display for DisplayBlock<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let indent = self.indent;
        for insn_id in self.block.insns.iter() {
            let insn = self.function.insn_at(*insn_id);
            if insn.has_output() {
                let ty = self.function.insn_types[insn_id.0].clone();
                // TODO(max): Figure out how to get `indent' worth of spaces
                write!(f, "  {insn_id:<indent$}:{ty} = {insn}\n")?;
            } else {
                write!(f, "  {insn}\n")?;
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for ManagedFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (idx, block) in self.blocks.iter().enumerate() {
            let display_block = DisplayBlock {
                function: self,
                block: &block,
                indent: 2,
            };
            write!(f, "bb {idx} ")?;
            if block.num_params() > 0 {
                write!(f, "(")?;
                let mut sep = "";
                for (idx, param_type) in block.param_types.iter().enumerate() {
                    write!(f, "{sep}arg{idx}: {param_type}")?;
                    sep = ", ";
                }
                write!(f, ") ")?;
            }
            write!(f, "{{\n{display_block}}}\n")?;
        }
        Ok(())
    }
}

fn sample_function() -> ManagedFunction {
    let mut result = ManagedFunction::new();
    let add = result.push(
        result.entrypoint,
        Insn::Add(Opnd::Const(Value::Int(3)), Opnd::Const(Value::Int(4))),
    );
    let lt = result.push(
        result.entrypoint,
        Insn::Lt(Opnd::InsnOut(add), Opnd::Const(Value::Int(8))),
    );
    let conseq = result.alloc_block();
    let alt = result.alloc_block();
    let ift = result.push(
        result.entrypoint,
        Insn::IfTrue(
            Opnd::InsnOut(lt),
            JumpEdge {
                target: conseq,
                opnds: vec![],
            },
            JumpEdge {
                target: alt,
                opnds: vec![],
            },
        ),
    );
    let join = result.alloc_block_with_params(1);
    result.push(
        conseq,
        Insn::Jump(JumpEdge {
            target: join,
            opnds: vec![Opnd::Const(Value::Str("hello".into()))],
        }),
    );
    result.push(
        alt,
        Insn::Jump(JumpEdge {
            target: join,
            opnds: vec![Opnd::Const(Value::Int(6))],
        }),
    );
    let add2 = result.push(join, Insn::Add(Opnd::Param(0), Opnd::Const(Value::Int(7))));
    result.push(
        join,
        Insn::Send {
            receiver: Opnd::InsnOut(add2),
            name: Opnd::Const(Value::Symbol("abs".into())),
            args: vec![],
        },
    );
    result.push(join, Insn::Return(Opnd::InsnOut(add2)));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let fun = sample_function();
        ()
    }
}

// TODO: do we want a struct to represent programs?
#[derive(Default, Debug)]
struct Program {
    // Permanent home of every class; grows without bound
    // Maps ClassId to class objects
    classes: Vec<ClassDesc>,

    // Permanent home of every function
    funs: Vec<Function>,

    // Main/entry function
    main: FunId,
}

impl Program {
    // TODO: pre-register types for things like Nil, Integer, etc?

    // Register a class
    pub fn reg_class(&mut self, ty: ClassDesc) -> ClassId {
        let result = ClassId(self.classes.len());
        self.classes.push(ty);
        result
    }

    pub fn reg_native_fun(&mut self, fun: NativeFunction) -> FunId {
        let result = FunId(self.funs.len());
        self.funs.push(Function::Native(fun));
        result
    }
}

pub struct LCG {
    state: u64,
    // Parameters from "Numerical Recipes"
    a: u64, // multiplier
    c: u64, // increment
    m: u64, // modulus (2^64 in this case)
}

impl LCG {
    pub fn new(seed: u64) -> Self {
        LCG {
            state: seed,
            a: 6364136223846793005,
            c: 1442695040888963407,
            m: u64::MAX, // 2^64 - 1
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
    pub fn next_bool(&mut self) -> bool {
        (self.next() & (1 << 60)) != 0
    }

    // Choose a random index in [0, max[
    pub fn next_idx(&mut self, max: usize) -> usize {
        (self.next_u32() as usize) % max
    }

    // Returns true with a specific percentage of probability
    pub fn pct_prob(&mut self, percent: usize) -> bool {
        let idx = self.next_idx(100);
        idx < percent
    }

    // Pick a random element from a slice
    pub fn choice<'a, T>(&mut self, slice: &'a [T]) -> &'a T {
        assert!(!slice.is_empty());
        &slice[self.next_idx(slice.len())]
    }
}

// TODO: we need a function to generate a big graph/program that's going to be
// a torture test
//
// Something like 10-20K classes and 20K methods that randomly call each other
// We want the size of it to approximate the size of our production apps
//
// IIRC the average size of a Ruby method is something like 8 bytecode instructions
// So we can generate many simple methods/functions
fn gen_torture_test(num_classes: usize, num_methods: usize) -> Program {
    let mut rng = LCG::new(0);

    let mut prog = Program::default();

    let mut class_ids = Vec::new();

    for i in 0..num_classes {
        let class_id = prog.reg_class(ClassDesc {
            name: format!("class_{i}"),
            fields: vec![],
            methods: HashMap::new(),
            ctor: FunId(0), /* TODO: need ctor method id? */
        });
        class_ids.push(class_id);
    }

    //let mut fun_ids = Vec::new();

    // Generate functions that only call previously defined functions.
    // This effectively creates a DAG of function calls, which we know
    // by construction can't have infinite recursion
    for i in 0..num_methods {
        // Leaf function returning a constant
        let mut fun = ManagedFunction::new();
        let block = fun.alloc_block();
        fun.push(block, Insn::Return(Opnd::Const(Value::Int(7))));
        fun.push(block, Insn::Return(Opnd::Const(Value::Int(2))));

        // TODO: need some way to assign function ids
        // Functions live on the program object, like classes?

        // TODO: we need some way to add/register methods to classes
        // TODO: assign the methods to random classes?
    }

    prog
}

const INT_TYPE: ClassId = ClassId(0);
const STR_TYPE: ClassId = ClassId(1);
const TRUE_TYPE: ClassId = ClassId(2);
const FALSE_TYPE: ClassId = ClassId(3);
const NIL_TYPE: ClassId = ClassId(4);

fn main() {
    let mut program = Program::default();
    let int_ctor = program.reg_native_fun(NativeFunction("Integer.new".into()));
    let str_ctor = program.reg_native_fun(NativeFunction("String.new".into()));
    let true_ctor = program.reg_native_fun(NativeFunction("TrueClass.new".into()));
    let false_ctor = program.reg_native_fun(NativeFunction("FalseClass.new".into()));
    let nil_ctor = program.reg_native_fun(NativeFunction("NilClass.new".into()));
    program.reg_class(ClassDesc {
        name: "Integer".into(),
        fields: vec![],
        methods: HashMap::new(),
        ctor: int_ctor,
    });
    program.reg_class(ClassDesc {
        name: "String".into(),
        fields: vec![],
        methods: HashMap::new(),
        ctor: str_ctor,
    });
    program.reg_class(ClassDesc {
        name: "TrueClass".into(),
        fields: vec![],
        methods: HashMap::new(),
        ctor: true_ctor,
    });
    program.reg_class(ClassDesc {
        name: "FalseClass".into(),
        fields: vec![],
        methods: HashMap::new(),
        ctor: false_ctor,
    });
    program.reg_class(ClassDesc {
        name: "NilClass".into(),
        fields: vec![],
        methods: HashMap::new(),
        ctor: nil_ctor,
    });
    let mut function = sample_function();
    function.reflow_types();
    println!("{function}");

    // TODO: run the analysis
    //
    // Generate a synthetic program and run the type analysis on it
    let prog = gen_torture_test(500, 2000);
    //prog.run_analysis();
}
