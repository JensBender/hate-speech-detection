лМ)
Ёё
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
Ѕ
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
А
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleщшelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleщшelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintџџџџџџџџџ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8б'
Б
,Adam/simple_rnn_17/simple_rnn_cell_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/simple_rnn_17/simple_rnn_cell_17/bias/v
Њ
@Adam/simple_rnn_17/simple_rnn_cell_17/bias/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_17/simple_rnn_cell_17/bias/v*
_output_shapes	
:*
dtype0
Ю
8Adam/simple_rnn_17/simple_rnn_cell_17/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*I
shared_name:8Adam/simple_rnn_17/simple_rnn_cell_17/recurrent_kernel/v
Ч
LAdam/simple_rnn_17/simple_rnn_cell_17/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_17/simple_rnn_cell_17/recurrent_kernel/v* 
_output_shapes
:
*
dtype0
К
.Adam/simple_rnn_17/simple_rnn_cell_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*?
shared_name0.Adam/simple_rnn_17/simple_rnn_cell_17/kernel/v
Г
BAdam/simple_rnn_17/simple_rnn_cell_17/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_17/simple_rnn_cell_17/kernel/v* 
_output_shapes
:
*
dtype0
Б
,Adam/simple_rnn_16/simple_rnn_cell_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/simple_rnn_16/simple_rnn_cell_16/bias/v
Њ
@Adam/simple_rnn_16/simple_rnn_cell_16/bias/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_16/simple_rnn_cell_16/bias/v*
_output_shapes	
:*
dtype0
Ю
8Adam/simple_rnn_16/simple_rnn_cell_16/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*I
shared_name:8Adam/simple_rnn_16/simple_rnn_cell_16/recurrent_kernel/v
Ч
LAdam/simple_rnn_16/simple_rnn_cell_16/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_16/simple_rnn_cell_16/recurrent_kernel/v* 
_output_shapes
:
*
dtype0
Й
.Adam/simple_rnn_16/simple_rnn_cell_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*?
shared_name0.Adam/simple_rnn_16/simple_rnn_cell_16/kernel/v
В
BAdam/simple_rnn_16/simple_rnn_cell_16/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_16/simple_rnn_cell_16/kernel/v*
_output_shapes
:	2*
dtype0

Adam/dense_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_25/bias/v
y
(Adam/dense_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/v*
_output_shapes
:*
dtype0

Adam/dense_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_25/kernel/v

*Adam/dense_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/v*
_output_shapes

:@*
dtype0

Adam/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_24/bias/v
y
(Adam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*'
shared_nameAdam/dense_24/kernel/v

*Adam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/v*
_output_shapes
:	@*
dtype0

Adam/embedding_12/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'2*/
shared_name Adam/embedding_12/embeddings/v

2Adam/embedding_12/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_12/embeddings/v*
_output_shapes
:	'2*
dtype0
Б
,Adam/simple_rnn_17/simple_rnn_cell_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/simple_rnn_17/simple_rnn_cell_17/bias/m
Њ
@Adam/simple_rnn_17/simple_rnn_cell_17/bias/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_17/simple_rnn_cell_17/bias/m*
_output_shapes	
:*
dtype0
Ю
8Adam/simple_rnn_17/simple_rnn_cell_17/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*I
shared_name:8Adam/simple_rnn_17/simple_rnn_cell_17/recurrent_kernel/m
Ч
LAdam/simple_rnn_17/simple_rnn_cell_17/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_17/simple_rnn_cell_17/recurrent_kernel/m* 
_output_shapes
:
*
dtype0
К
.Adam/simple_rnn_17/simple_rnn_cell_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*?
shared_name0.Adam/simple_rnn_17/simple_rnn_cell_17/kernel/m
Г
BAdam/simple_rnn_17/simple_rnn_cell_17/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_17/simple_rnn_cell_17/kernel/m* 
_output_shapes
:
*
dtype0
Б
,Adam/simple_rnn_16/simple_rnn_cell_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/simple_rnn_16/simple_rnn_cell_16/bias/m
Њ
@Adam/simple_rnn_16/simple_rnn_cell_16/bias/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_16/simple_rnn_cell_16/bias/m*
_output_shapes	
:*
dtype0
Ю
8Adam/simple_rnn_16/simple_rnn_cell_16/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*I
shared_name:8Adam/simple_rnn_16/simple_rnn_cell_16/recurrent_kernel/m
Ч
LAdam/simple_rnn_16/simple_rnn_cell_16/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_16/simple_rnn_cell_16/recurrent_kernel/m* 
_output_shapes
:
*
dtype0
Й
.Adam/simple_rnn_16/simple_rnn_cell_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*?
shared_name0.Adam/simple_rnn_16/simple_rnn_cell_16/kernel/m
В
BAdam/simple_rnn_16/simple_rnn_cell_16/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_16/simple_rnn_cell_16/kernel/m*
_output_shapes
:	2*
dtype0

Adam/dense_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_25/bias/m
y
(Adam/dense_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/m*
_output_shapes
:*
dtype0

Adam/dense_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_25/kernel/m

*Adam/dense_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/m*
_output_shapes

:@*
dtype0

Adam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_24/bias/m
y
(Adam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*'
shared_nameAdam/dense_24/kernel/m

*Adam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/m*
_output_shapes
:	@*
dtype0

Adam/embedding_12/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'2*/
shared_name Adam/embedding_12/embeddings/m

2Adam/embedding_12/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_12/embeddings/m*
_output_shapes
:	'2*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
Ѓ
%simple_rnn_17/simple_rnn_cell_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%simple_rnn_17/simple_rnn_cell_17/bias

9simple_rnn_17/simple_rnn_cell_17/bias/Read/ReadVariableOpReadVariableOp%simple_rnn_17/simple_rnn_cell_17/bias*
_output_shapes	
:*
dtype0
Р
1simple_rnn_17/simple_rnn_cell_17/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*B
shared_name31simple_rnn_17/simple_rnn_cell_17/recurrent_kernel
Й
Esimple_rnn_17/simple_rnn_cell_17/recurrent_kernel/Read/ReadVariableOpReadVariableOp1simple_rnn_17/simple_rnn_cell_17/recurrent_kernel* 
_output_shapes
:
*
dtype0
Ќ
'simple_rnn_17/simple_rnn_cell_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*8
shared_name)'simple_rnn_17/simple_rnn_cell_17/kernel
Ѕ
;simple_rnn_17/simple_rnn_cell_17/kernel/Read/ReadVariableOpReadVariableOp'simple_rnn_17/simple_rnn_cell_17/kernel* 
_output_shapes
:
*
dtype0
Ѓ
%simple_rnn_16/simple_rnn_cell_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%simple_rnn_16/simple_rnn_cell_16/bias

9simple_rnn_16/simple_rnn_cell_16/bias/Read/ReadVariableOpReadVariableOp%simple_rnn_16/simple_rnn_cell_16/bias*
_output_shapes	
:*
dtype0
Р
1simple_rnn_16/simple_rnn_cell_16/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*B
shared_name31simple_rnn_16/simple_rnn_cell_16/recurrent_kernel
Й
Esimple_rnn_16/simple_rnn_cell_16/recurrent_kernel/Read/ReadVariableOpReadVariableOp1simple_rnn_16/simple_rnn_cell_16/recurrent_kernel* 
_output_shapes
:
*
dtype0
Ћ
'simple_rnn_16/simple_rnn_cell_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*8
shared_name)'simple_rnn_16/simple_rnn_cell_16/kernel
Є
;simple_rnn_16/simple_rnn_cell_16/kernel/Read/ReadVariableOpReadVariableOp'simple_rnn_16/simple_rnn_cell_16/kernel*
_output_shapes
:	2*
dtype0
r
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_25/bias
k
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes
:*
dtype0
z
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_25/kernel
s
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel*
_output_shapes

:@*
dtype0
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
:@*
dtype0
{
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_namedense_24/kernel
t
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes
:	@*
dtype0

embedding_12/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'2*(
shared_nameembedding_12/embeddings

+embedding_12/embeddings/Read/ReadVariableOpReadVariableOpembedding_12/embeddings*
_output_shapes
:	'2*
dtype0

"serving_default_embedding_12_inputPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
В
StatefulPartitionedCallStatefulPartitionedCall"serving_default_embedding_12_inputembedding_12/embeddings'simple_rnn_16/simple_rnn_cell_16/kernel%simple_rnn_16/simple_rnn_cell_16/bias1simple_rnn_16/simple_rnn_cell_16/recurrent_kernel'simple_rnn_17/simple_rnn_cell_17/kernel%simple_rnn_17/simple_rnn_cell_17/bias1simple_rnn_17/simple_rnn_cell_17/recurrent_kerneldense_24/kerneldense_24/biasdense_25/kerneldense_25/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_356709

NoOpNoOp
вV
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*V
valueVBV BљU

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
 
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*
Њ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec*
Њ
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$cell
%
state_spec*
І
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias*
І
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias*
R
0
61
72
83
94
:5
;6
,7
-8
49
510*
R
0
61
72
83
94
:5
;6
,7
-8
49
510*
* 
А
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Atrace_0
Btrace_1
Ctrace_2
Dtrace_3* 
6
Etrace_0
Ftrace_1
Gtrace_2
Htrace_3* 
* 
 
Iiter

Jbeta_1

Kbeta_2
	Ldecay
Mlearning_ratemЋ,mЌ-m­4mЎ5mЏ6mА7mБ8mВ9mГ:mД;mЕvЖ,vЗ-vИ4vЙ5vК6vЛ7vМ8vН9vО:vП;vР*

Nserving_default* 

0*

0*
* 

Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ttrace_0* 

Utrace_0* 
ke
VARIABLE_VALUEembedding_12/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

60
71
82*

60
71
82*
* 


Vstates
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
\trace_0
]trace_1
^trace_2
_trace_3* 
6
`trace_0
atrace_1
btrace_2
ctrace_3* 
г
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses
j_random_generator

6kernel
7recurrent_kernel
8bias*
* 

90
:1
;2*

90
:1
;2*
* 


kstates
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
6
qtrace_0
rtrace_1
strace_2
ttrace_3* 
6
utrace_0
vtrace_1
wtrace_2
xtrace_3* 
г
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
_random_generator

9kernel
:recurrent_kernel
;bias*
* 

,0
-1*

,0
-1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

trace_0* 

trace_0* 
_Y
VARIABLE_VALUEdense_24/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_24/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

40
51*

40
51*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

trace_0* 

trace_0* 
_Y
VARIABLE_VALUEdense_25/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_25/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'simple_rnn_16/simple_rnn_cell_16/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1simple_rnn_16/simple_rnn_cell_16/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_16/simple_rnn_cell_16/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'simple_rnn_17/simple_rnn_cell_17/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1simple_rnn_17/simple_rnn_cell_17/recurrent_kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_17/simple_rnn_cell_17/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

60
71
82*

60
71
82*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
* 
* 

$0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

90
:1
;2*

90
:1
;2*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*

trace_0
trace_1* 

 trace_0
Ёtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
Ђ	variables
Ѓ	keras_api

Єtotal

Ѕcount*
M
І	variables
Ї	keras_api

Јtotal

Љcount
Њ
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Є0
Ѕ1*

Ђ	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Ј0
Љ1*

І	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

VARIABLE_VALUEAdam/embedding_12/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_24/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_24/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_25/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_25/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.Adam/simple_rnn_16/simple_rnn_cell_16/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/simple_rnn_16/simple_rnn_cell_16/recurrent_kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/simple_rnn_16/simple_rnn_cell_16/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.Adam/simple_rnn_17/simple_rnn_cell_17/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/simple_rnn_17/simple_rnn_cell_17/recurrent_kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/simple_rnn_17/simple_rnn_cell_17/bias/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/embedding_12/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_24/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_24/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_25/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_25/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.Adam/simple_rnn_16/simple_rnn_cell_16/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/simple_rnn_16/simple_rnn_cell_16/recurrent_kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/simple_rnn_16/simple_rnn_cell_16/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.Adam/simple_rnn_17/simple_rnn_cell_17/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/simple_rnn_17/simple_rnn_cell_17/recurrent_kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/simple_rnn_17/simple_rnn_cell_17/bias/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Љ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+embedding_12/embeddings/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOp;simple_rnn_16/simple_rnn_cell_16/kernel/Read/ReadVariableOpEsimple_rnn_16/simple_rnn_cell_16/recurrent_kernel/Read/ReadVariableOp9simple_rnn_16/simple_rnn_cell_16/bias/Read/ReadVariableOp;simple_rnn_17/simple_rnn_cell_17/kernel/Read/ReadVariableOpEsimple_rnn_17/simple_rnn_cell_17/recurrent_kernel/Read/ReadVariableOp9simple_rnn_17/simple_rnn_cell_17/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp2Adam/embedding_12/embeddings/m/Read/ReadVariableOp*Adam/dense_24/kernel/m/Read/ReadVariableOp(Adam/dense_24/bias/m/Read/ReadVariableOp*Adam/dense_25/kernel/m/Read/ReadVariableOp(Adam/dense_25/bias/m/Read/ReadVariableOpBAdam/simple_rnn_16/simple_rnn_cell_16/kernel/m/Read/ReadVariableOpLAdam/simple_rnn_16/simple_rnn_cell_16/recurrent_kernel/m/Read/ReadVariableOp@Adam/simple_rnn_16/simple_rnn_cell_16/bias/m/Read/ReadVariableOpBAdam/simple_rnn_17/simple_rnn_cell_17/kernel/m/Read/ReadVariableOpLAdam/simple_rnn_17/simple_rnn_cell_17/recurrent_kernel/m/Read/ReadVariableOp@Adam/simple_rnn_17/simple_rnn_cell_17/bias/m/Read/ReadVariableOp2Adam/embedding_12/embeddings/v/Read/ReadVariableOp*Adam/dense_24/kernel/v/Read/ReadVariableOp(Adam/dense_24/bias/v/Read/ReadVariableOp*Adam/dense_25/kernel/v/Read/ReadVariableOp(Adam/dense_25/bias/v/Read/ReadVariableOpBAdam/simple_rnn_16/simple_rnn_cell_16/kernel/v/Read/ReadVariableOpLAdam/simple_rnn_16/simple_rnn_cell_16/recurrent_kernel/v/Read/ReadVariableOp@Adam/simple_rnn_16/simple_rnn_cell_16/bias/v/Read/ReadVariableOpBAdam/simple_rnn_17/simple_rnn_cell_17/kernel/v/Read/ReadVariableOpLAdam/simple_rnn_17/simple_rnn_cell_17/recurrent_kernel/v/Read/ReadVariableOp@Adam/simple_rnn_17/simple_rnn_cell_17/bias/v/Read/ReadVariableOpConst*7
Tin0
.2,	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_359173
м
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_12/embeddingsdense_24/kerneldense_24/biasdense_25/kerneldense_25/bias'simple_rnn_16/simple_rnn_cell_16/kernel1simple_rnn_16/simple_rnn_cell_16/recurrent_kernel%simple_rnn_16/simple_rnn_cell_16/bias'simple_rnn_17/simple_rnn_cell_17/kernel1simple_rnn_17/simple_rnn_cell_17/recurrent_kernel%simple_rnn_17/simple_rnn_cell_17/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/embedding_12/embeddings/mAdam/dense_24/kernel/mAdam/dense_24/bias/mAdam/dense_25/kernel/mAdam/dense_25/bias/m.Adam/simple_rnn_16/simple_rnn_cell_16/kernel/m8Adam/simple_rnn_16/simple_rnn_cell_16/recurrent_kernel/m,Adam/simple_rnn_16/simple_rnn_cell_16/bias/m.Adam/simple_rnn_17/simple_rnn_cell_17/kernel/m8Adam/simple_rnn_17/simple_rnn_cell_17/recurrent_kernel/m,Adam/simple_rnn_17/simple_rnn_cell_17/bias/mAdam/embedding_12/embeddings/vAdam/dense_24/kernel/vAdam/dense_24/bias/vAdam/dense_25/kernel/vAdam/dense_25/bias/v.Adam/simple_rnn_16/simple_rnn_cell_16/kernel/v8Adam/simple_rnn_16/simple_rnn_cell_16/recurrent_kernel/v,Adam/simple_rnn_16/simple_rnn_cell_16/bias/v.Adam/simple_rnn_17/simple_rnn_cell_17/kernel/v8Adam/simple_rnn_17/simple_rnn_cell_17/recurrent_kernel/v,Adam/simple_rnn_17/simple_rnn_cell_17/bias/v*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_359309№Њ%
!
р
while_body_355074
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
!while_simple_rnn_cell_16_355096_0:	20
!while_simple_rnn_cell_16_355098_0:	5
!while_simple_rnn_cell_16_355100_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
while_simple_rnn_cell_16_355096:	2.
while_simple_rnn_cell_16_355098:	3
while_simple_rnn_cell_16_355100:
Ђ0while/simple_rnn_cell_16/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ2*
element_dtype0Љ
0while/simple_rnn_cell_16/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_16_355096_0!while_simple_rnn_cell_16_355098_0!while_simple_rnn_cell_16_355100_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_355061т
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_16/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity9while/simple_rnn_cell_16/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџ

while/NoOpNoOp1^while/simple_rnn_cell_16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_16_355096!while_simple_rnn_cell_16_355096_0"D
while_simple_rnn_cell_16_355098!while_simple_rnn_cell_16_355098_0"D
while_simple_rnn_cell_16_355100!while_simple_rnn_cell_16_355100_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџ: : : : : 2d
0while/simple_rnn_cell_16/StatefulPartitionedCall0while/simple_rnn_cell_16/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 

№
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_358983

inputs
states_02
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	4
 matmul_1_readvariableop_resource:

identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџI
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
mulMulinputsones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0k
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ_
mul_1Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
MatMul_1MatMul	mul_1:z:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџH
TanhTanhadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџX
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџZ

Identity_1IdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
Ў
Џ
I__inference_sequential_12_layer_call_and_return_conditional_losses_357459

inputs7
$embedding_12_embedding_lookup_357083:	'2R
?simple_rnn_16_simple_rnn_cell_16_matmul_readvariableop_resource:	2O
@simple_rnn_16_simple_rnn_cell_16_biasadd_readvariableop_resource:	U
Asimple_rnn_16_simple_rnn_cell_16_matmul_1_readvariableop_resource:
S
?simple_rnn_17_simple_rnn_cell_17_matmul_readvariableop_resource:
O
@simple_rnn_17_simple_rnn_cell_17_biasadd_readvariableop_resource:	U
Asimple_rnn_17_simple_rnn_cell_17_matmul_1_readvariableop_resource:
:
'dense_24_matmul_readvariableop_resource:	@6
(dense_24_biasadd_readvariableop_resource:@9
'dense_25_matmul_readvariableop_resource:@6
(dense_25_biasadd_readvariableop_resource:
identityЂdense_24/BiasAdd/ReadVariableOpЂdense_24/MatMul/ReadVariableOpЂdense_25/BiasAdd/ReadVariableOpЂdense_25/MatMul/ReadVariableOpЂembedding_12/embedding_lookupЂ7simple_rnn_16/simple_rnn_cell_16/BiasAdd/ReadVariableOpЂ6simple_rnn_16/simple_rnn_cell_16/MatMul/ReadVariableOpЂ8simple_rnn_16/simple_rnn_cell_16/MatMul_1/ReadVariableOpЂsimple_rnn_16/whileЂ7simple_rnn_17/simple_rnn_cell_17/BiasAdd/ReadVariableOpЂ6simple_rnn_17/simple_rnn_cell_17/MatMul/ReadVariableOpЂ8simple_rnn_17/simple_rnn_cell_17/MatMul_1/ReadVariableOpЂsimple_rnn_17/whileb
embedding_12/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџя
embedding_12/embedding_lookupResourceGather$embedding_12_embedding_lookup_357083embedding_12/Cast:y:0*
Tindices0*7
_class-
+)loc:@embedding_12/embedding_lookup/357083*+
_output_shapes
:џџџџџџџџџ2*
dtype0Щ
&embedding_12/embedding_lookup/IdentityIdentity&embedding_12/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_12/embedding_lookup/357083*+
_output_shapes
:џџџџџџџџџ2
(embedding_12/embedding_lookup/Identity_1Identity/embedding_12/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2\
embedding_12/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
embedding_12/NotEqualNotEqualinputs embedding_12/NotEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџt
simple_rnn_16/ShapeShape1embedding_12/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:k
!simple_rnn_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_16/strided_sliceStridedSlicesimple_rnn_16/Shape:output:0*simple_rnn_16/strided_slice/stack:output:0,simple_rnn_16/strided_slice/stack_1:output:0,simple_rnn_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
simple_rnn_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
simple_rnn_16/zeros/packedPack$simple_rnn_16/strided_slice:output:0%simple_rnn_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
simple_rnn_16/zerosFill#simple_rnn_16/zeros/packed:output:0"simple_rnn_16/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџq
simple_rnn_16/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Д
simple_rnn_16/transpose	Transpose1embedding_12/embedding_lookup/Identity_1:output:0%simple_rnn_16/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2`
simple_rnn_16/Shape_1Shapesimple_rnn_16/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
simple_rnn_16/strided_slice_1StridedSlicesimple_rnn_16/Shape_1:output:0,simple_rnn_16/strided_slice_1/stack:output:0.simple_rnn_16/strided_slice_1/stack_1:output:0.simple_rnn_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
simple_rnn_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
simple_rnn_16/ExpandDims
ExpandDimsembedding_12/NotEqual:z:0%simple_rnn_16/ExpandDims/dim:output:0*
T0
*+
_output_shapes
:џџџџџџџџџs
simple_rnn_16/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ј
simple_rnn_16/transpose_1	Transpose!simple_rnn_16/ExpandDims:output:0'simple_rnn_16/transpose_1/perm:output:0*
T0
*+
_output_shapes
:џџџџџџџџџt
)simple_rnn_16/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџо
simple_rnn_16/TensorArrayV2TensorListReserve2simple_rnn_16/TensorArrayV2/element_shape:output:0&simple_rnn_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Csimple_rnn_16/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   
5simple_rnn_16/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_16/transpose:y:0Lsimple_rnn_16/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвm
#simple_rnn_16/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_16/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_16/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Џ
simple_rnn_16/strided_slice_2StridedSlicesimple_rnn_16/transpose:y:0,simple_rnn_16/strided_slice_2/stack:output:0.simple_rnn_16/strided_slice_2/stack_1:output:0.simple_rnn_16/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask
0simple_rnn_16/simple_rnn_cell_16/ones_like/ShapeShape&simple_rnn_16/strided_slice_2:output:0*
T0*
_output_shapes
:u
0simple_rnn_16/simple_rnn_cell_16/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?к
*simple_rnn_16/simple_rnn_cell_16/ones_likeFill9simple_rnn_16/simple_rnn_cell_16/ones_like/Shape:output:09simple_rnn_16/simple_rnn_cell_16/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2s
.simple_rnn_16/simple_rnn_cell_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @г
,simple_rnn_16/simple_rnn_cell_16/dropout/MulMul3simple_rnn_16/simple_rnn_cell_16/ones_like:output:07simple_rnn_16/simple_rnn_cell_16/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
.simple_rnn_16/simple_rnn_cell_16/dropout/ShapeShape3simple_rnn_16/simple_rnn_cell_16/ones_like:output:0*
T0*
_output_shapes
:Ю
Esimple_rnn_16/simple_rnn_cell_16/dropout/random_uniform/RandomUniformRandomUniform7simple_rnn_16/simple_rnn_cell_16/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype0|
7simple_rnn_16/simple_rnn_cell_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
5simple_rnn_16/simple_rnn_cell_16/dropout/GreaterEqualGreaterEqualNsimple_rnn_16/simple_rnn_cell_16/dropout/random_uniform/RandomUniform:output:0@simple_rnn_16/simple_rnn_cell_16/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2Б
-simple_rnn_16/simple_rnn_cell_16/dropout/CastCast9simple_rnn_16/simple_rnn_cell_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2Ь
.simple_rnn_16/simple_rnn_cell_16/dropout/Mul_1Mul0simple_rnn_16/simple_rnn_cell_16/dropout/Mul:z:01simple_rnn_16/simple_rnn_cell_16/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2~
2simple_rnn_16/simple_rnn_cell_16/ones_like_1/ShapeShapesimple_rnn_16/zeros:output:0*
T0*
_output_shapes
:w
2simple_rnn_16/simple_rnn_cell_16/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?с
,simple_rnn_16/simple_rnn_cell_16/ones_like_1Fill;simple_rnn_16/simple_rnn_cell_16/ones_like_1/Shape:output:0;simple_rnn_16/simple_rnn_cell_16/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџu
0simple_rnn_16/simple_rnn_cell_16/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @к
.simple_rnn_16/simple_rnn_cell_16/dropout_1/MulMul5simple_rnn_16/simple_rnn_cell_16/ones_like_1:output:09simple_rnn_16/simple_rnn_cell_16/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
0simple_rnn_16/simple_rnn_cell_16/dropout_1/ShapeShape5simple_rnn_16/simple_rnn_cell_16/ones_like_1:output:0*
T0*
_output_shapes
:г
Gsimple_rnn_16/simple_rnn_cell_16/dropout_1/random_uniform/RandomUniformRandomUniform9simple_rnn_16/simple_rnn_cell_16/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0~
9simple_rnn_16/simple_rnn_cell_16/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
7simple_rnn_16/simple_rnn_cell_16/dropout_1/GreaterEqualGreaterEqualPsimple_rnn_16/simple_rnn_cell_16/dropout_1/random_uniform/RandomUniform:output:0Bsimple_rnn_16/simple_rnn_cell_16/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЖ
/simple_rnn_16/simple_rnn_cell_16/dropout_1/CastCast;simple_rnn_16/simple_rnn_cell_16/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџг
0simple_rnn_16/simple_rnn_cell_16/dropout_1/Mul_1Mul2simple_rnn_16/simple_rnn_cell_16/dropout_1/Mul:z:03simple_rnn_16/simple_rnn_cell_16/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЙ
$simple_rnn_16/simple_rnn_cell_16/mulMul&simple_rnn_16/strided_slice_2:output:02simple_rnn_16/simple_rnn_cell_16/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2З
6simple_rnn_16/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp?simple_rnn_16_simple_rnn_cell_16_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype0Ю
'simple_rnn_16/simple_rnn_cell_16/MatMulMatMul(simple_rnn_16/simple_rnn_cell_16/mul:z:0>simple_rnn_16/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЕ
7simple_rnn_16/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_16_simple_rnn_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0к
(simple_rnn_16/simple_rnn_cell_16/BiasAddBiasAdd1simple_rnn_16/simple_rnn_cell_16/MatMul:product:0?simple_rnn_16/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџД
&simple_rnn_16/simple_rnn_cell_16/mul_1Mulsimple_rnn_16/zeros:output:04simple_rnn_16/simple_rnn_cell_16/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџМ
8simple_rnn_16/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_16_simple_rnn_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0д
)simple_rnn_16/simple_rnn_cell_16/MatMul_1MatMul*simple_rnn_16/simple_rnn_cell_16/mul_1:z:0@simple_rnn_16/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ
$simple_rnn_16/simple_rnn_cell_16/addAddV21simple_rnn_16/simple_rnn_cell_16/BiasAdd:output:03simple_rnn_16/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
%simple_rnn_16/simple_rnn_cell_16/TanhTanh(simple_rnn_16/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ|
+simple_rnn_16/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   т
simple_rnn_16/TensorArrayV2_1TensorListReserve4simple_rnn_16/TensorArrayV2_1/element_shape:output:0&simple_rnn_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвT
simple_rnn_16/timeConst*
_output_shapes
: *
dtype0*
value	B : v
+simple_rnn_16/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџт
simple_rnn_16/TensorArrayV2_2TensorListReserve4simple_rnn_16/TensorArrayV2_2/element_shape:output:0&simple_rnn_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
Esimple_rnn_16/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
7simple_rnn_16/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorsimple_rnn_16/transpose_1:y:0Nsimple_rnn_16/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
simple_rnn_16/zeros_like	ZerosLike)simple_rnn_16/simple_rnn_cell_16/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџq
&simple_rnn_16/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџb
 simple_rnn_16/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ј
simple_rnn_16/whileWhile)simple_rnn_16/while/loop_counter:output:0/simple_rnn_16/while/maximum_iterations:output:0simple_rnn_16/time:output:0&simple_rnn_16/TensorArrayV2_1:handle:0simple_rnn_16/zeros_like:y:0simple_rnn_16/zeros:output:0&simple_rnn_16/strided_slice_1:output:0Esimple_rnn_16/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gsimple_rnn_16/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0?simple_rnn_16_simple_rnn_cell_16_matmul_readvariableop_resource@simple_rnn_16_simple_rnn_cell_16_biasadd_readvariableop_resourceAsimple_rnn_16_simple_rnn_cell_16_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*P
_output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *+
body#R!
simple_rnn_16_while_body_357162*+
cond#R!
simple_rnn_16_while_cond_357161*O
output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *
parallel_iterations 
>simple_rnn_16/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   э
0simple_rnn_16/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_16/while:output:3Gsimple_rnn_16/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0v
#simple_rnn_16/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџo
%simple_rnn_16/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_16/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ю
simple_rnn_16/strided_slice_3StridedSlice9simple_rnn_16/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_16/strided_slice_3/stack:output:0.simple_rnn_16/strided_slice_3/stack_1:output:0.simple_rnn_16/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_masks
simple_rnn_16/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          С
simple_rnn_16/transpose_2	Transpose9simple_rnn_16/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_16/transpose_2/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ`
simple_rnn_17/ShapeShapesimple_rnn_16/transpose_2:y:0*
T0*
_output_shapes
:k
!simple_rnn_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_17/strided_sliceStridedSlicesimple_rnn_17/Shape:output:0*simple_rnn_17/strided_slice/stack:output:0,simple_rnn_17/strided_slice/stack_1:output:0,simple_rnn_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
simple_rnn_17/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
simple_rnn_17/zeros/packedPack$simple_rnn_17/strided_slice:output:0%simple_rnn_17/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_17/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
simple_rnn_17/zerosFill#simple_rnn_17/zeros/packed:output:0"simple_rnn_17/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџq
simple_rnn_17/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ё
simple_rnn_17/transpose	Transposesimple_rnn_16/transpose_2:y:0%simple_rnn_17/transpose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ`
simple_rnn_17/Shape_1Shapesimple_rnn_17/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
simple_rnn_17/strided_slice_1StridedSlicesimple_rnn_17/Shape_1:output:0,simple_rnn_17/strided_slice_1/stack:output:0.simple_rnn_17/strided_slice_1/stack_1:output:0.simple_rnn_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
simple_rnn_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
simple_rnn_17/ExpandDims
ExpandDimsembedding_12/NotEqual:z:0%simple_rnn_17/ExpandDims/dim:output:0*
T0
*+
_output_shapes
:џџџџџџџџџs
simple_rnn_17/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ј
simple_rnn_17/transpose_1	Transpose!simple_rnn_17/ExpandDims:output:0'simple_rnn_17/transpose_1/perm:output:0*
T0
*+
_output_shapes
:џџџџџџџџџt
)simple_rnn_17/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџо
simple_rnn_17/TensorArrayV2TensorListReserve2simple_rnn_17/TensorArrayV2/element_shape:output:0&simple_rnn_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Csimple_rnn_17/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
5simple_rnn_17/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_17/transpose:y:0Lsimple_rnn_17/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвm
#simple_rnn_17/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_17/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_17/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
simple_rnn_17/strided_slice_2StridedSlicesimple_rnn_17/transpose:y:0,simple_rnn_17/strided_slice_2/stack:output:0.simple_rnn_17/strided_slice_2/stack_1:output:0.simple_rnn_17/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
0simple_rnn_17/simple_rnn_cell_17/ones_like/ShapeShape&simple_rnn_17/strided_slice_2:output:0*
T0*
_output_shapes
:u
0simple_rnn_17/simple_rnn_cell_17/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?л
*simple_rnn_17/simple_rnn_cell_17/ones_likeFill9simple_rnn_17/simple_rnn_cell_17/ones_like/Shape:output:09simple_rnn_17/simple_rnn_cell_17/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџs
.simple_rnn_17/simple_rnn_cell_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @д
,simple_rnn_17/simple_rnn_cell_17/dropout/MulMul3simple_rnn_17/simple_rnn_cell_17/ones_like:output:07simple_rnn_17/simple_rnn_cell_17/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
.simple_rnn_17/simple_rnn_cell_17/dropout/ShapeShape3simple_rnn_17/simple_rnn_cell_17/ones_like:output:0*
T0*
_output_shapes
:Я
Esimple_rnn_17/simple_rnn_cell_17/dropout/random_uniform/RandomUniformRandomUniform7simple_rnn_17/simple_rnn_cell_17/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0|
7simple_rnn_17/simple_rnn_cell_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
5simple_rnn_17/simple_rnn_cell_17/dropout/GreaterEqualGreaterEqualNsimple_rnn_17/simple_rnn_cell_17/dropout/random_uniform/RandomUniform:output:0@simple_rnn_17/simple_rnn_cell_17/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџВ
-simple_rnn_17/simple_rnn_cell_17/dropout/CastCast9simple_rnn_17/simple_rnn_cell_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЭ
.simple_rnn_17/simple_rnn_cell_17/dropout/Mul_1Mul0simple_rnn_17/simple_rnn_cell_17/dropout/Mul:z:01simple_rnn_17/simple_rnn_cell_17/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ~
2simple_rnn_17/simple_rnn_cell_17/ones_like_1/ShapeShapesimple_rnn_17/zeros:output:0*
T0*
_output_shapes
:w
2simple_rnn_17/simple_rnn_cell_17/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?с
,simple_rnn_17/simple_rnn_cell_17/ones_like_1Fill;simple_rnn_17/simple_rnn_cell_17/ones_like_1/Shape:output:0;simple_rnn_17/simple_rnn_cell_17/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџu
0simple_rnn_17/simple_rnn_cell_17/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @к
.simple_rnn_17/simple_rnn_cell_17/dropout_1/MulMul5simple_rnn_17/simple_rnn_cell_17/ones_like_1:output:09simple_rnn_17/simple_rnn_cell_17/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
0simple_rnn_17/simple_rnn_cell_17/dropout_1/ShapeShape5simple_rnn_17/simple_rnn_cell_17/ones_like_1:output:0*
T0*
_output_shapes
:г
Gsimple_rnn_17/simple_rnn_cell_17/dropout_1/random_uniform/RandomUniformRandomUniform9simple_rnn_17/simple_rnn_cell_17/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0~
9simple_rnn_17/simple_rnn_cell_17/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
7simple_rnn_17/simple_rnn_cell_17/dropout_1/GreaterEqualGreaterEqualPsimple_rnn_17/simple_rnn_cell_17/dropout_1/random_uniform/RandomUniform:output:0Bsimple_rnn_17/simple_rnn_cell_17/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЖ
/simple_rnn_17/simple_rnn_cell_17/dropout_1/CastCast;simple_rnn_17/simple_rnn_cell_17/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџг
0simple_rnn_17/simple_rnn_cell_17/dropout_1/Mul_1Mul2simple_rnn_17/simple_rnn_cell_17/dropout_1/Mul:z:03simple_rnn_17/simple_rnn_cell_17/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџК
$simple_rnn_17/simple_rnn_cell_17/mulMul&simple_rnn_17/strided_slice_2:output:02simple_rnn_17/simple_rnn_cell_17/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџИ
6simple_rnn_17/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp?simple_rnn_17_simple_rnn_cell_17_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ю
'simple_rnn_17/simple_rnn_cell_17/MatMulMatMul(simple_rnn_17/simple_rnn_cell_17/mul:z:0>simple_rnn_17/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЕ
7simple_rnn_17/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_17_simple_rnn_cell_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0к
(simple_rnn_17/simple_rnn_cell_17/BiasAddBiasAdd1simple_rnn_17/simple_rnn_cell_17/MatMul:product:0?simple_rnn_17/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџД
&simple_rnn_17/simple_rnn_cell_17/mul_1Mulsimple_rnn_17/zeros:output:04simple_rnn_17/simple_rnn_cell_17/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџМ
8simple_rnn_17/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_17_simple_rnn_cell_17_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0д
)simple_rnn_17/simple_rnn_cell_17/MatMul_1MatMul*simple_rnn_17/simple_rnn_cell_17/mul_1:z:0@simple_rnn_17/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ
$simple_rnn_17/simple_rnn_cell_17/addAddV21simple_rnn_17/simple_rnn_cell_17/BiasAdd:output:03simple_rnn_17/simple_rnn_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
%simple_rnn_17/simple_rnn_cell_17/TanhTanh(simple_rnn_17/simple_rnn_cell_17/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ|
+simple_rnn_17/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   l
*simple_rnn_17/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :я
simple_rnn_17/TensorArrayV2_1TensorListReserve4simple_rnn_17/TensorArrayV2_1/element_shape:output:03simple_rnn_17/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвT
simple_rnn_17/timeConst*
_output_shapes
: *
dtype0*
value	B : v
+simple_rnn_17/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџт
simple_rnn_17/TensorArrayV2_2TensorListReserve4simple_rnn_17/TensorArrayV2_2/element_shape:output:0&simple_rnn_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
Esimple_rnn_17/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
7simple_rnn_17/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorsimple_rnn_17/transpose_1:y:0Nsimple_rnn_17/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
simple_rnn_17/zeros_like	ZerosLike)simple_rnn_17/simple_rnn_cell_17/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџq
&simple_rnn_17/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџb
 simple_rnn_17/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ј
simple_rnn_17/whileWhile)simple_rnn_17/while/loop_counter:output:0/simple_rnn_17/while/maximum_iterations:output:0simple_rnn_17/time:output:0&simple_rnn_17/TensorArrayV2_1:handle:0simple_rnn_17/zeros_like:y:0simple_rnn_17/zeros:output:0&simple_rnn_17/strided_slice_1:output:0Esimple_rnn_17/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gsimple_rnn_17/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0?simple_rnn_17_simple_rnn_cell_17_matmul_readvariableop_resource@simple_rnn_17_simple_rnn_cell_17_biasadd_readvariableop_resourceAsimple_rnn_17_simple_rnn_cell_17_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*P
_output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *+
body#R!
simple_rnn_17_while_body_357339*+
cond#R!
simple_rnn_17_while_cond_357338*O
output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *
parallel_iterations 
>simple_rnn_17/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
0simple_rnn_17/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_17/while:output:3Gsimple_rnn_17/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsv
#simple_rnn_17/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџo
%simple_rnn_17/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_17/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ю
simple_rnn_17/strided_slice_3StridedSlice9simple_rnn_17/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_17/strided_slice_3/stack:output:0.simple_rnn_17/strided_slice_3/stack_1:output:0.simple_rnn_17/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_masks
simple_rnn_17/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          С
simple_rnn_17/transpose_2	Transpose9simple_rnn_17/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_17/transpose_2/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_24/MatMulMatMul&simple_rnn_17/strided_slice_3:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@b
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_25/MatMulMatMuldense_24/Relu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџh
dense_25/SigmoidSigmoiddense_25/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
IdentityIdentitydense_25/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџє
NoOpNoOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp^embedding_12/embedding_lookup8^simple_rnn_16/simple_rnn_cell_16/BiasAdd/ReadVariableOp7^simple_rnn_16/simple_rnn_cell_16/MatMul/ReadVariableOp9^simple_rnn_16/simple_rnn_cell_16/MatMul_1/ReadVariableOp^simple_rnn_16/while8^simple_rnn_17/simple_rnn_cell_17/BiasAdd/ReadVariableOp7^simple_rnn_17/simple_rnn_cell_17/MatMul/ReadVariableOp9^simple_rnn_17/simple_rnn_cell_17/MatMul_1/ReadVariableOp^simple_rnn_17/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2>
embedding_12/embedding_lookupembedding_12/embedding_lookup2r
7simple_rnn_16/simple_rnn_cell_16/BiasAdd/ReadVariableOp7simple_rnn_16/simple_rnn_cell_16/BiasAdd/ReadVariableOp2p
6simple_rnn_16/simple_rnn_cell_16/MatMul/ReadVariableOp6simple_rnn_16/simple_rnn_cell_16/MatMul/ReadVariableOp2t
8simple_rnn_16/simple_rnn_cell_16/MatMul_1/ReadVariableOp8simple_rnn_16/simple_rnn_cell_16/MatMul_1/ReadVariableOp2*
simple_rnn_16/whilesimple_rnn_16/while2r
7simple_rnn_17/simple_rnn_cell_17/BiasAdd/ReadVariableOp7simple_rnn_17/simple_rnn_cell_17/BiasAdd/ReadVariableOp2p
6simple_rnn_17/simple_rnn_cell_17/MatMul/ReadVariableOp6simple_rnn_17/simple_rnn_cell_17/MatMul/ReadVariableOp2t
8simple_rnn_17/simple_rnn_cell_17/MatMul_1/ReadVariableOp8simple_rnn_17/simple_rnn_cell_17/MatMul_1/ReadVariableOp2*
simple_rnn_17/whilesimple_rnn_17/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


ѕ
D__inference_dense_25_layer_call_and_return_conditional_losses_358836

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ѕ	
І
H__inference_embedding_12_layer_call_and_return_conditional_losses_355674

inputs*
embedding_lookup_355668:	'2
identityЂembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџЛ
embedding_lookupResourceGatherembedding_lookup_355668Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/355668*+
_output_shapes
:џџџџџџџџџ2*
dtype0Ђ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/355668*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЖQ
а
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_357951

inputs
mask
D
1simple_rnn_cell_16_matmul_readvariableop_resource:	2A
2simple_rnn_cell_16_biasadd_readvariableop_resource:	G
3simple_rnn_cell_16_matmul_1_readvariableop_resource:

identityЂ)simple_rnn_cell_16/BiasAdd/ReadVariableOpЂ(simple_rnn_cell_16/MatMul/ReadVariableOpЂ*simple_rnn_cell_16/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџm

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*+
_output_shapes
:џџџџџџџџџe
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ~
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*+
_output_shapes
:џџџџџџџџџf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maskj
"simple_rnn_cell_16/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:g
"simple_rnn_cell_16/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?А
simple_rnn_cell_16/ones_likeFill+simple_rnn_cell_16/ones_like/Shape:output:0+simple_rnn_cell_16/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2b
$simple_rnn_cell_16/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:i
$simple_rnn_cell_16/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?З
simple_rnn_cell_16/ones_like_1Fill-simple_rnn_cell_16/ones_like_1/Shape:output:0-simple_rnn_cell_16/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_16/mulMulstrided_slice_2:output:0%simple_rnn_cell_16/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
(simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_16_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype0Є
simple_rnn_cell_16/MatMulMatMulsimple_rnn_cell_16/mul:z:00simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
)simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0А
simple_rnn_cell_16/BiasAddBiasAdd#simple_rnn_cell_16/MatMul:product:01simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_16/mul_1Mulzeros:output:0'simple_rnn_cell_16/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
*simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Њ
simple_rnn_cell_16/MatMul_1MatMulsimple_rnn_cell_16/mul_1:z:02simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_16/addAddV2#simple_rnn_cell_16/BiasAdd:output:0%simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџn
simple_rnn_cell_16/TanhTanhsimple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : h
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџИ
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ц
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШg

zeros_like	ZerosLikesimple_rnn_cell_16/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџc
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ж
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:01simple_rnn_cell_16_matmul_readvariableop_resource2simple_rnn_cell_16_biasadd_readvariableop_resource3simple_rnn_cell_16_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*P
_output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_357862*
condR
while_cond_357861*O
output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   У
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџc
IdentityIdentitytranspose_2:y:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџв
NoOpNoOp*^simple_rnn_cell_16/BiasAdd/ReadVariableOp)^simple_rnn_cell_16/MatMul/ReadVariableOp+^simple_rnn_cell_16/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџ2:џџџџџџџџџ: : : 2V
)simple_rnn_cell_16/BiasAdd/ReadVariableOp)simple_rnn_cell_16/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_16/MatMul/ReadVariableOp(simple_rnn_cell_16/MatMul/ReadVariableOp2X
*simple_rnn_cell_16/MatMul_1/ReadVariableOp*simple_rnn_cell_16/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs:MI
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask
і[
Т

while_body_356166
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0M
9while_simple_rnn_cell_17_matmul_readvariableop_resource_0:
I
:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0:	O
;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorK
7while_simple_rnn_cell_17_matmul_readvariableop_resource:
G
8while_simple_rnn_cell_17_biasadd_readvariableop_resource:	M
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource:
Ђ/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpЂ.while/simple_rnn_cell_17/MatMul/ReadVariableOpЂ0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ў
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0

(while/simple_rnn_cell_17/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(while/simple_rnn_cell_17/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?У
"while/simple_rnn_cell_17/ones_likeFill1while/simple_rnn_cell_17/ones_like/Shape:output:01while/simple_rnn_cell_17/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџk
&while/simple_rnn_cell_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @М
$while/simple_rnn_cell_17/dropout/MulMul+while/simple_rnn_cell_17/ones_like:output:0/while/simple_rnn_cell_17/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
&while/simple_rnn_cell_17/dropout/ShapeShape+while/simple_rnn_cell_17/ones_like:output:0*
T0*
_output_shapes
:П
=while/simple_rnn_cell_17/dropout/random_uniform/RandomUniformRandomUniform/while/simple_rnn_cell_17/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0t
/while/simple_rnn_cell_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ђ
-while/simple_rnn_cell_17/dropout/GreaterEqualGreaterEqualFwhile/simple_rnn_cell_17/dropout/random_uniform/RandomUniform:output:08while/simple_rnn_cell_17/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЂ
%while/simple_rnn_cell_17/dropout/CastCast1while/simple_rnn_cell_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЕ
&while/simple_rnn_cell_17/dropout/Mul_1Mul(while/simple_rnn_cell_17/dropout/Mul:z:0)while/simple_rnn_cell_17/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџm
*while/simple_rnn_cell_17/ones_like_1/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:o
*while/simple_rnn_cell_17/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Щ
$while/simple_rnn_cell_17/ones_like_1Fill3while/simple_rnn_cell_17/ones_like_1/Shape:output:03while/simple_rnn_cell_17/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџm
(while/simple_rnn_cell_17/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Т
&while/simple_rnn_cell_17/dropout_1/MulMul-while/simple_rnn_cell_17/ones_like_1:output:01while/simple_rnn_cell_17/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
(while/simple_rnn_cell_17/dropout_1/ShapeShape-while/simple_rnn_cell_17/ones_like_1:output:0*
T0*
_output_shapes
:У
?while/simple_rnn_cell_17/dropout_1/random_uniform/RandomUniformRandomUniform1while/simple_rnn_cell_17/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0v
1while/simple_rnn_cell_17/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ј
/while/simple_rnn_cell_17/dropout_1/GreaterEqualGreaterEqualHwhile/simple_rnn_cell_17/dropout_1/random_uniform/RandomUniform:output:0:while/simple_rnn_cell_17/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџІ
'while/simple_rnn_cell_17/dropout_1/CastCast3while/simple_rnn_cell_17/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЛ
(while/simple_rnn_cell_17/dropout_1/Mul_1Mul*while/simple_rnn_cell_17/dropout_1/Mul:z:0+while/simple_rnn_cell_17/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџД
while/simple_rnn_cell_17/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/simple_rnn_cell_17/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЊ
.while/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_17_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ж
while/simple_rnn_cell_17/MatMulMatMul while/simple_rnn_cell_17/mul:z:06while/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЇ
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Т
 while/simple_rnn_cell_17/BiasAddBiasAdd)while/simple_rnn_cell_17/MatMul:product:07while/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/simple_rnn_cell_17/mul_1Mulwhile_placeholder_3,while/simple_rnn_cell_17/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЎ
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0М
!while/simple_rnn_cell_17/MatMul_1MatMul"while/simple_rnn_cell_17/mul_1:z:08while/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџА
while/simple_rnn_cell_17/addAddV2)while/simple_rnn_cell_17/BiasAdd:output:0+while/simple_rnn_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџz
while/simple_rnn_cell_17/TanhTanh while/simple_rnn_cell_17/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџe
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
while/SelectV2SelectV2while/Tile:output:0!while/simple_rnn_cell_17/Tanh:y:0while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџg
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
while/SelectV2_1SelectV2while/Tile_1:output:0!while/simple_rnn_cell_17/Tanh:y:0while_placeholder_3*
T0*(
_output_shapes
:џџџџџџџџџr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ш
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: u
while/Identity_4Identitywhile/SelectV2:output:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
while/Identity_5Identitywhile/SelectV2_1:output:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџт

while/NoOpNoOp0^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_simple_rnn_cell_17_biasadd_readvariableop_resource:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_17_matmul_readvariableop_resource9while_simple_rnn_cell_17_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"А
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : 2b
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_17/MatMul/ReadVariableOp.while/simple_rnn_cell_17/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ФR
б
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_355986

inputs
mask
E
1simple_rnn_cell_17_matmul_readvariableop_resource:
A
2simple_rnn_cell_17_biasadd_readvariableop_resource:	G
3simple_rnn_cell_17_matmul_1_readvariableop_resource:

identityЂ)simple_rnn_cell_17/BiasAdd/ReadVariableOpЂ(simple_rnn_cell_17/MatMul/ReadVariableOpЂ*simple_rnn_cell_17/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџm

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*+
_output_shapes
:џџџџџџџџџe
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ~
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*+
_output_shapes
:џџџџџџџџџf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskj
"simple_rnn_cell_17/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:g
"simple_rnn_cell_17/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Б
simple_rnn_cell_17/ones_likeFill+simple_rnn_cell_17/ones_like/Shape:output:0+simple_rnn_cell_17/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
$simple_rnn_cell_17/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:i
$simple_rnn_cell_17/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?З
simple_rnn_cell_17/ones_like_1Fill-simple_rnn_cell_17/ones_like_1/Shape:output:0-simple_rnn_cell_17/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/mulMulstrided_slice_2:output:0%simple_rnn_cell_17/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
(simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_17_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Є
simple_rnn_cell_17/MatMulMatMulsimple_rnn_cell_17/mul:z:00simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
)simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0А
simple_rnn_cell_17/BiasAddBiasAdd#simple_rnn_cell_17/MatMul:product:01simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/mul_1Mulzeros:output:0'simple_rnn_cell_17/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
*simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_17_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Њ
simple_rnn_cell_17/MatMul_1MatMulsimple_rnn_cell_17/mul_1:z:02simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/addAddV2#simple_rnn_cell_17/BiasAdd:output:0%simple_rnn_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџn
simple_rnn_cell_17/TanhTanhsimple_rnn_cell_17/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : h
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџИ
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ц
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШg

zeros_like	ZerosLikesimple_rnn_cell_17/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџc
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ж
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:01simple_rnn_cell_17_matmul_readvariableop_resource2simple_rnn_cell_17_biasadd_readvariableop_resource3simple_rnn_cell_17_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*P
_output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_355896*
condR
while_cond_355895*O
output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   з
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџв
NoOpNoOp*^simple_rnn_cell_17/BiasAdd/ReadVariableOp)^simple_rnn_cell_17/MatMul/ReadVariableOp+^simple_rnn_cell_17/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:џџџџџџџџџ:џџџџџџџџџ: : : 2V
)simple_rnn_cell_17/BiasAdd/ReadVariableOp)simple_rnn_cell_17/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_17/MatMul/ReadVariableOp(simple_rnn_cell_17/MatMul/ReadVariableOp2X
*simple_rnn_cell_17/MatMul_1/ReadVariableOp*simple_rnn_cell_17/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:MI
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask
	
љ
while_cond_358689
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_358689___redundant_placeholder04
0while_while_cond_358689___redundant_placeholder14
0while_while_cond_358689___redundant_placeholder24
0while_while_cond_358689___redundant_placeholder34
0while_while_cond_358689___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
И$
э
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_355205

inputs

states1
matmul_readvariableop_resource:	2.
biasadd_readvariableop_resource:	4
 matmul_1_readvariableop_resource:

identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2G
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @w
dropout_1/MulMulones_like_1:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџS
dropout_1/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџt
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџp
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџW
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2*
dtype0k
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ\
mul_1Mulstatesdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
MatMul_1MatMul	mul_1:z:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџH
TanhTanhadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџX
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџZ

Identity_1IdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџ2:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs:PL
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates
Щ

Ё
$__inference_signature_wrapper_356709
embedding_12_input
unknown:	'2
	unknown_0:	2
	unknown_1:	
	unknown_2:

	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	@
	unknown_7:@
	unknown_8:@
	unknown_9:
identityЂStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallembedding_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_355005o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameembedding_12_input
Й"
т
while_body_355584
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
!while_simple_rnn_cell_17_355606_0:
0
!while_simple_rnn_cell_17_355608_0:	5
!while_simple_rnn_cell_17_355610_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
while_simple_rnn_cell_17_355606:
.
while_simple_rnn_cell_17_355608:	3
while_simple_rnn_cell_17_355610:
Ђ0while/simple_rnn_cell_17/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0Љ
0while/simple_rnn_cell_17/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_17_355606_0!while_simple_rnn_cell_17_355608_0!while_simple_rnn_cell_17_355610_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_355531r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:09while/simple_rnn_cell_17/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity9while/simple_rnn_cell_17/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџ

while/NoOpNoOp1^while/simple_rnn_cell_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_17_355606!while_simple_rnn_cell_17_355606_0"D
while_simple_rnn_cell_17_355608!while_simple_rnn_cell_17_355608_0"D
while_simple_rnn_cell_17_355610!while_simple_rnn_cell_17_355610_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџ: : : : : 2d
0while/simple_rnn_cell_17/StatefulPartitionedCall0while/simple_rnn_cell_17/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
м
Њ
while_cond_355073
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_355073___redundant_placeholder04
0while_while_cond_355073___redundant_placeholder14
0while_while_cond_355073___redundant_placeholder24
0while_while_cond_355073___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
тF
Т

while_body_355896
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0M
9while_simple_rnn_cell_17_matmul_readvariableop_resource_0:
I
:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0:	O
;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorK
7while_simple_rnn_cell_17_matmul_readvariableop_resource:
G
8while_simple_rnn_cell_17_biasadd_readvariableop_resource:	M
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource:
Ђ/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpЂ.while/simple_rnn_cell_17/MatMul/ReadVariableOpЂ0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ў
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0

(while/simple_rnn_cell_17/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(while/simple_rnn_cell_17/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?У
"while/simple_rnn_cell_17/ones_likeFill1while/simple_rnn_cell_17/ones_like/Shape:output:01while/simple_rnn_cell_17/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџm
*while/simple_rnn_cell_17/ones_like_1/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:o
*while/simple_rnn_cell_17/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Щ
$while/simple_rnn_cell_17/ones_like_1Fill3while/simple_rnn_cell_17/ones_like_1/Shape:output:03while/simple_rnn_cell_17/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЕ
while/simple_rnn_cell_17/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/simple_rnn_cell_17/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЊ
.while/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_17_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ж
while/simple_rnn_cell_17/MatMulMatMul while/simple_rnn_cell_17/mul:z:06while/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЇ
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Т
 while/simple_rnn_cell_17/BiasAddBiasAdd)while/simple_rnn_cell_17/MatMul:product:07while/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/simple_rnn_cell_17/mul_1Mulwhile_placeholder_3-while/simple_rnn_cell_17/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЎ
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0М
!while/simple_rnn_cell_17/MatMul_1MatMul"while/simple_rnn_cell_17/mul_1:z:08while/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџА
while/simple_rnn_cell_17/addAddV2)while/simple_rnn_cell_17/BiasAdd:output:0+while/simple_rnn_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџz
while/simple_rnn_cell_17/TanhTanh while/simple_rnn_cell_17/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџe
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
while/SelectV2SelectV2while/Tile:output:0!while/simple_rnn_cell_17/Tanh:y:0while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџg
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
while/SelectV2_1SelectV2while/Tile_1:output:0!while/simple_rnn_cell_17/Tanh:y:0while_placeholder_3*
T0*(
_output_shapes
:џџџџџџџџџr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ш
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: u
while/Identity_4Identitywhile/SelectV2:output:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
while/Identity_5Identitywhile/SelectV2_1:output:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџт

while/NoOpNoOp0^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_simple_rnn_cell_17_biasadd_readvariableop_resource:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_17_matmul_readvariableop_resource9while_simple_rnn_cell_17_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"А
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : 2b
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_17/MatMul/ReadVariableOp.while/simple_rnn_cell_17/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
з


.__inference_sequential_12_layer_call_fn_356763

inputs
unknown:	'2
	unknown_0:	2
	unknown_1:	
	unknown_2:

	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	@
	unknown_7:@
	unknown_8:@
	unknown_9:
identityЂStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_356556o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Z
Ш
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_357802
inputs_0D
1simple_rnn_cell_16_matmul_readvariableop_resource:	2A
2simple_rnn_cell_16_biasadd_readvariableop_resource:	G
3simple_rnn_cell_16_matmul_1_readvariableop_resource:

identityЂ)simple_rnn_cell_16/BiasAdd/ReadVariableOpЂ(simple_rnn_cell_16/MatMul/ReadVariableOpЂ*simple_rnn_cell_16/MatMul_1/ReadVariableOpЂwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maskj
"simple_rnn_cell_16/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:g
"simple_rnn_cell_16/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?А
simple_rnn_cell_16/ones_likeFill+simple_rnn_cell_16/ones_like/Shape:output:0+simple_rnn_cell_16/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2e
 simple_rnn_cell_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Љ
simple_rnn_cell_16/dropout/MulMul%simple_rnn_cell_16/ones_like:output:0)simple_rnn_cell_16/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2u
 simple_rnn_cell_16/dropout/ShapeShape%simple_rnn_cell_16/ones_like:output:0*
T0*
_output_shapes
:В
7simple_rnn_cell_16/dropout/random_uniform/RandomUniformRandomUniform)simple_rnn_cell_16/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype0n
)simple_rnn_cell_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?п
'simple_rnn_cell_16/dropout/GreaterEqualGreaterEqual@simple_rnn_cell_16/dropout/random_uniform/RandomUniform:output:02simple_rnn_cell_16/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_16/dropout/CastCast+simple_rnn_cell_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2Ђ
 simple_rnn_cell_16/dropout/Mul_1Mul"simple_rnn_cell_16/dropout/Mul:z:0#simple_rnn_cell_16/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2b
$simple_rnn_cell_16/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:i
$simple_rnn_cell_16/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?З
simple_rnn_cell_16/ones_like_1Fill-simple_rnn_cell_16/ones_like_1/Shape:output:0-simple_rnn_cell_16/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџg
"simple_rnn_cell_16/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @А
 simple_rnn_cell_16/dropout_1/MulMul'simple_rnn_cell_16/ones_like_1:output:0+simple_rnn_cell_16/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџy
"simple_rnn_cell_16/dropout_1/ShapeShape'simple_rnn_cell_16/ones_like_1:output:0*
T0*
_output_shapes
:З
9simple_rnn_cell_16/dropout_1/random_uniform/RandomUniformRandomUniform+simple_rnn_cell_16/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0p
+simple_rnn_cell_16/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ц
)simple_rnn_cell_16/dropout_1/GreaterEqualGreaterEqualBsimple_rnn_cell_16/dropout_1/random_uniform/RandomUniform:output:04simple_rnn_cell_16/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
!simple_rnn_cell_16/dropout_1/CastCast-simple_rnn_cell_16/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЉ
"simple_rnn_cell_16/dropout_1/Mul_1Mul$simple_rnn_cell_16/dropout_1/Mul:z:0%simple_rnn_cell_16/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_16/mulMulstrided_slice_2:output:0$simple_rnn_cell_16/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
(simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_16_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype0Є
simple_rnn_cell_16/MatMulMatMulsimple_rnn_cell_16/mul:z:00simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
)simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0А
simple_rnn_cell_16/BiasAddBiasAdd#simple_rnn_cell_16/MatMul:product:01simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_16/mul_1Mulzeros:output:0&simple_rnn_cell_16/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ 
*simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Њ
simple_rnn_cell_16/MatMul_1MatMulsimple_rnn_cell_16/mul_1:z:02simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_16/addAddV2#simple_rnn_cell_16/BiasAdd:output:0%simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџn
simple_rnn_cell_16/TanhTanhsimple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_16_matmul_readvariableop_resource2simple_rnn_cell_16_biasadd_readvariableop_resource3simple_rnn_cell_16_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_357712*
condR
while_cond_357711*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ь
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџв
NoOpNoOp*^simple_rnn_cell_16/BiasAdd/ReadVariableOp)^simple_rnn_cell_16/MatMul/ReadVariableOp+^simple_rnn_cell_16/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ2: : : 2V
)simple_rnn_cell_16/BiasAdd/ReadVariableOp)simple_rnn_cell_16/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_16/MatMul/ReadVariableOp(simple_rnn_cell_16/MatMul/ReadVariableOp2X
*simple_rnn_cell_16/MatMul_1/ReadVariableOp*simple_rnn_cell_16/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
"
_user_specified_name
inputs/0
єr
К
simple_rnn_17_while_body_3573398
4simple_rnn_17_while_simple_rnn_17_while_loop_counter>
:simple_rnn_17_while_simple_rnn_17_while_maximum_iterations#
simple_rnn_17_while_placeholder%
!simple_rnn_17_while_placeholder_1%
!simple_rnn_17_while_placeholder_2%
!simple_rnn_17_while_placeholder_37
3simple_rnn_17_while_simple_rnn_17_strided_slice_1_0s
osimple_rnn_17_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_tensorlistfromtensor_0w
ssimple_rnn_17_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_1_tensorlistfromtensor_0[
Gsimple_rnn_17_while_simple_rnn_cell_17_matmul_readvariableop_resource_0:
W
Hsimple_rnn_17_while_simple_rnn_cell_17_biasadd_readvariableop_resource_0:	]
Isimple_rnn_17_while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0:
 
simple_rnn_17_while_identity"
simple_rnn_17_while_identity_1"
simple_rnn_17_while_identity_2"
simple_rnn_17_while_identity_3"
simple_rnn_17_while_identity_4"
simple_rnn_17_while_identity_55
1simple_rnn_17_while_simple_rnn_17_strided_slice_1q
msimple_rnn_17_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_tensorlistfromtensoru
qsimple_rnn_17_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_1_tensorlistfromtensorY
Esimple_rnn_17_while_simple_rnn_cell_17_matmul_readvariableop_resource:
U
Fsimple_rnn_17_while_simple_rnn_cell_17_biasadd_readvariableop_resource:	[
Gsimple_rnn_17_while_simple_rnn_cell_17_matmul_1_readvariableop_resource:
Ђ=simple_rnn_17/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpЂ<simple_rnn_17/while/simple_rnn_cell_17/MatMul/ReadVariableOpЂ>simple_rnn_17/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp
Esimple_rnn_17/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   э
7simple_rnn_17/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_17_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_17_while_placeholderNsimple_rnn_17/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
Gsimple_rnn_17/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   є
9simple_rnn_17/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemssimple_rnn_17_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_1_tensorlistfromtensor_0simple_rnn_17_while_placeholderPsimple_rnn_17/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
Є
6simple_rnn_17/while/simple_rnn_cell_17/ones_like/ShapeShape>simple_rnn_17/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:{
6simple_rnn_17/while/simple_rnn_cell_17/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?э
0simple_rnn_17/while/simple_rnn_cell_17/ones_likeFill?simple_rnn_17/while/simple_rnn_cell_17/ones_like/Shape:output:0?simple_rnn_17/while/simple_rnn_cell_17/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџy
4simple_rnn_17/while/simple_rnn_cell_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @ц
2simple_rnn_17/while/simple_rnn_cell_17/dropout/MulMul9simple_rnn_17/while/simple_rnn_cell_17/ones_like:output:0=simple_rnn_17/while/simple_rnn_cell_17/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
4simple_rnn_17/while/simple_rnn_cell_17/dropout/ShapeShape9simple_rnn_17/while/simple_rnn_cell_17/ones_like:output:0*
T0*
_output_shapes
:л
Ksimple_rnn_17/while/simple_rnn_cell_17/dropout/random_uniform/RandomUniformRandomUniform=simple_rnn_17/while/simple_rnn_cell_17/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0
=simple_rnn_17/while/simple_rnn_cell_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
;simple_rnn_17/while/simple_rnn_cell_17/dropout/GreaterEqualGreaterEqualTsimple_rnn_17/while/simple_rnn_cell_17/dropout/random_uniform/RandomUniform:output:0Fsimple_rnn_17/while/simple_rnn_cell_17/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџО
3simple_rnn_17/while/simple_rnn_cell_17/dropout/CastCast?simple_rnn_17/while/simple_rnn_cell_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџп
4simple_rnn_17/while/simple_rnn_cell_17/dropout/Mul_1Mul6simple_rnn_17/while/simple_rnn_cell_17/dropout/Mul:z:07simple_rnn_17/while/simple_rnn_cell_17/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
8simple_rnn_17/while/simple_rnn_cell_17/ones_like_1/ShapeShape!simple_rnn_17_while_placeholder_3*
T0*
_output_shapes
:}
8simple_rnn_17/while/simple_rnn_cell_17/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ѓ
2simple_rnn_17/while/simple_rnn_cell_17/ones_like_1FillAsimple_rnn_17/while/simple_rnn_cell_17/ones_like_1/Shape:output:0Asimple_rnn_17/while/simple_rnn_cell_17/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ{
6simple_rnn_17/while/simple_rnn_cell_17/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @ь
4simple_rnn_17/while/simple_rnn_cell_17/dropout_1/MulMul;simple_rnn_17/while/simple_rnn_cell_17/ones_like_1:output:0?simple_rnn_17/while/simple_rnn_cell_17/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
6simple_rnn_17/while/simple_rnn_cell_17/dropout_1/ShapeShape;simple_rnn_17/while/simple_rnn_cell_17/ones_like_1:output:0*
T0*
_output_shapes
:п
Msimple_rnn_17/while/simple_rnn_cell_17/dropout_1/random_uniform/RandomUniformRandomUniform?simple_rnn_17/while/simple_rnn_cell_17/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0
?simple_rnn_17/while/simple_rnn_cell_17/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ђ
=simple_rnn_17/while/simple_rnn_cell_17/dropout_1/GreaterEqualGreaterEqualVsimple_rnn_17/while/simple_rnn_cell_17/dropout_1/random_uniform/RandomUniform:output:0Hsimple_rnn_17/while/simple_rnn_cell_17/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџТ
5simple_rnn_17/while/simple_rnn_cell_17/dropout_1/CastCastAsimple_rnn_17/while/simple_rnn_cell_17/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџх
6simple_rnn_17/while/simple_rnn_cell_17/dropout_1/Mul_1Mul8simple_rnn_17/while/simple_rnn_cell_17/dropout_1/Mul:z:09simple_rnn_17/while/simple_rnn_cell_17/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџо
*simple_rnn_17/while/simple_rnn_cell_17/mulMul>simple_rnn_17/while/TensorArrayV2Read/TensorListGetItem:item:08simple_rnn_17/while/simple_rnn_cell_17/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЦ
<simple_rnn_17/while/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_17_while_simple_rnn_cell_17_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0р
-simple_rnn_17/while/simple_rnn_cell_17/MatMulMatMul.simple_rnn_17/while/simple_rnn_cell_17/mul:z:0Dsimple_rnn_17/while/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџУ
=simple_rnn_17/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_17_while_simple_rnn_cell_17_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ь
.simple_rnn_17/while/simple_rnn_cell_17/BiasAddBiasAdd7simple_rnn_17/while/simple_rnn_cell_17/MatMul:product:0Esimple_rnn_17/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ
,simple_rnn_17/while/simple_rnn_cell_17/mul_1Mul!simple_rnn_17_while_placeholder_3:simple_rnn_17/while/simple_rnn_cell_17/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЪ
>simple_rnn_17/while/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_17_while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0ц
/simple_rnn_17/while/simple_rnn_cell_17/MatMul_1MatMul0simple_rnn_17/while/simple_rnn_cell_17/mul_1:z:0Fsimple_rnn_17/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџк
*simple_rnn_17/while/simple_rnn_cell_17/addAddV27simple_rnn_17/while/simple_rnn_cell_17/BiasAdd:output:09simple_rnn_17/while/simple_rnn_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
+simple_rnn_17/while/simple_rnn_cell_17/TanhTanh.simple_rnn_17/while/simple_rnn_cell_17/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџs
"simple_rnn_17/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      С
simple_rnn_17/while/TileTile@simple_rnn_17/while/TensorArrayV2Read_1/TensorListGetItem:item:0+simple_rnn_17/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџв
simple_rnn_17/while/SelectV2SelectV2!simple_rnn_17/while/Tile:output:0/simple_rnn_17/while/simple_rnn_cell_17/Tanh:y:0!simple_rnn_17_while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџu
$simple_rnn_17/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      Х
simple_rnn_17/while/Tile_1Tile@simple_rnn_17/while/TensorArrayV2Read_1/TensorListGetItem:item:0-simple_rnn_17/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџж
simple_rnn_17/while/SelectV2_1SelectV2#simple_rnn_17/while/Tile_1:output:0/simple_rnn_17/while/simple_rnn_cell_17/Tanh:y:0!simple_rnn_17_while_placeholder_3*
T0*(
_output_shapes
:џџџџџџџџџ
>simple_rnn_17/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B :  
8simple_rnn_17/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_17_while_placeholder_1Gsimple_rnn_17/while/TensorArrayV2Write/TensorListSetItem/index:output:0%simple_rnn_17/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшв[
simple_rnn_17/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_17/while/addAddV2simple_rnn_17_while_placeholder"simple_rnn_17/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_17/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_17/while/add_1AddV24simple_rnn_17_while_simple_rnn_17_while_loop_counter$simple_rnn_17/while/add_1/y:output:0*
T0*
_output_shapes
: 
simple_rnn_17/while/IdentityIdentitysimple_rnn_17/while/add_1:z:0^simple_rnn_17/while/NoOp*
T0*
_output_shapes
: Ђ
simple_rnn_17/while/Identity_1Identity:simple_rnn_17_while_simple_rnn_17_while_maximum_iterations^simple_rnn_17/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_17/while/Identity_2Identitysimple_rnn_17/while/add:z:0^simple_rnn_17/while/NoOp*
T0*
_output_shapes
: А
simple_rnn_17/while/Identity_3IdentityHsimple_rnn_17/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_17/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_17/while/Identity_4Identity%simple_rnn_17/while/SelectV2:output:0^simple_rnn_17/while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџЁ
simple_rnn_17/while/Identity_5Identity'simple_rnn_17/while/SelectV2_1:output:0^simple_rnn_17/while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_17/while/NoOpNoOp>^simple_rnn_17/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp=^simple_rnn_17/while/simple_rnn_cell_17/MatMul/ReadVariableOp?^simple_rnn_17/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_17_while_identity%simple_rnn_17/while/Identity:output:0"I
simple_rnn_17_while_identity_1'simple_rnn_17/while/Identity_1:output:0"I
simple_rnn_17_while_identity_2'simple_rnn_17/while/Identity_2:output:0"I
simple_rnn_17_while_identity_3'simple_rnn_17/while/Identity_3:output:0"I
simple_rnn_17_while_identity_4'simple_rnn_17/while/Identity_4:output:0"I
simple_rnn_17_while_identity_5'simple_rnn_17/while/Identity_5:output:0"h
1simple_rnn_17_while_simple_rnn_17_strided_slice_13simple_rnn_17_while_simple_rnn_17_strided_slice_1_0"
Fsimple_rnn_17_while_simple_rnn_cell_17_biasadd_readvariableop_resourceHsimple_rnn_17_while_simple_rnn_cell_17_biasadd_readvariableop_resource_0"
Gsimple_rnn_17_while_simple_rnn_cell_17_matmul_1_readvariableop_resourceIsimple_rnn_17_while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0"
Esimple_rnn_17_while_simple_rnn_cell_17_matmul_readvariableop_resourceGsimple_rnn_17_while_simple_rnn_cell_17_matmul_readvariableop_resource_0"ш
qsimple_rnn_17_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_1_tensorlistfromtensorssimple_rnn_17_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_1_tensorlistfromtensor_0"р
msimple_rnn_17_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_tensorlistfromtensorosimple_rnn_17_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : 2~
=simple_rnn_17/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp=simple_rnn_17/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp2|
<simple_rnn_17/while/simple_rnn_cell_17/MatMul/ReadVariableOp<simple_rnn_17/while/simple_rnn_cell_17/MatMul/ReadVariableOp2
>simple_rnn_17/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp>simple_rnn_17/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
рL
л
while_body_358371
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0M
9while_simple_rnn_cell_17_matmul_readvariableop_resource_0:
I
:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0:	O
;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorK
7while_simple_rnn_cell_17_matmul_readvariableop_resource:
G
8while_simple_rnn_cell_17_biasadd_readvariableop_resource:	M
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource:
Ђ/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpЂ.while/simple_rnn_cell_17/MatMul/ReadVariableOpЂ0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
(while/simple_rnn_cell_17/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(while/simple_rnn_cell_17/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?У
"while/simple_rnn_cell_17/ones_likeFill1while/simple_rnn_cell_17/ones_like/Shape:output:01while/simple_rnn_cell_17/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџk
&while/simple_rnn_cell_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @М
$while/simple_rnn_cell_17/dropout/MulMul+while/simple_rnn_cell_17/ones_like:output:0/while/simple_rnn_cell_17/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
&while/simple_rnn_cell_17/dropout/ShapeShape+while/simple_rnn_cell_17/ones_like:output:0*
T0*
_output_shapes
:П
=while/simple_rnn_cell_17/dropout/random_uniform/RandomUniformRandomUniform/while/simple_rnn_cell_17/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0t
/while/simple_rnn_cell_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ђ
-while/simple_rnn_cell_17/dropout/GreaterEqualGreaterEqualFwhile/simple_rnn_cell_17/dropout/random_uniform/RandomUniform:output:08while/simple_rnn_cell_17/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЂ
%while/simple_rnn_cell_17/dropout/CastCast1while/simple_rnn_cell_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЕ
&while/simple_rnn_cell_17/dropout/Mul_1Mul(while/simple_rnn_cell_17/dropout/Mul:z:0)while/simple_rnn_cell_17/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџm
*while/simple_rnn_cell_17/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:o
*while/simple_rnn_cell_17/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Щ
$while/simple_rnn_cell_17/ones_like_1Fill3while/simple_rnn_cell_17/ones_like_1/Shape:output:03while/simple_rnn_cell_17/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџm
(while/simple_rnn_cell_17/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Т
&while/simple_rnn_cell_17/dropout_1/MulMul-while/simple_rnn_cell_17/ones_like_1:output:01while/simple_rnn_cell_17/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
(while/simple_rnn_cell_17/dropout_1/ShapeShape-while/simple_rnn_cell_17/ones_like_1:output:0*
T0*
_output_shapes
:У
?while/simple_rnn_cell_17/dropout_1/random_uniform/RandomUniformRandomUniform1while/simple_rnn_cell_17/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0v
1while/simple_rnn_cell_17/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ј
/while/simple_rnn_cell_17/dropout_1/GreaterEqualGreaterEqualHwhile/simple_rnn_cell_17/dropout_1/random_uniform/RandomUniform:output:0:while/simple_rnn_cell_17/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџІ
'while/simple_rnn_cell_17/dropout_1/CastCast3while/simple_rnn_cell_17/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЛ
(while/simple_rnn_cell_17/dropout_1/Mul_1Mul*while/simple_rnn_cell_17/dropout_1/Mul:z:0+while/simple_rnn_cell_17/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџД
while/simple_rnn_cell_17/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/simple_rnn_cell_17/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЊ
.while/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_17_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ж
while/simple_rnn_cell_17/MatMulMatMul while/simple_rnn_cell_17/mul:z:06while/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЇ
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Т
 while/simple_rnn_cell_17/BiasAddBiasAdd)while/simple_rnn_cell_17/MatMul:product:07while/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/simple_rnn_cell_17/mul_1Mulwhile_placeholder_2,while/simple_rnn_cell_17/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЎ
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0М
!while/simple_rnn_cell_17/MatMul_1MatMul"while/simple_rnn_cell_17/mul_1:z:08while/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџА
while/simple_rnn_cell_17/addAddV2)while/simple_rnn_cell_17/BiasAdd:output:0+while/simple_rnn_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџz
while/simple_rnn_cell_17/TanhTanh while/simple_rnn_cell_17/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ђ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_17/Tanh:y:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity!while/simple_rnn_cell_17/Tanh:y:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџт

while/NoOpNoOp0^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_17_biasadd_readvariableop_resource:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_17_matmul_readvariableop_resource9while_simple_rnn_cell_17_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџ: : : : : 2b
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_17/MatMul/ReadVariableOp.while/simple_rnn_cell_17/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
ЯZ
Р

while_body_358027
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0L
9while_simple_rnn_cell_16_matmul_readvariableop_resource_0:	2I
:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0:	O
;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorJ
7while_simple_rnn_cell_16_matmul_readvariableop_resource:	2G
8while_simple_rnn_cell_16_biasadd_readvariableop_resource:	M
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource:
Ђ/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpЂ.while/simple_rnn_cell_16/MatMul/ReadVariableOpЂ0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ2*
element_dtype0
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ў
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0

(while/simple_rnn_cell_16/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(while/simple_rnn_cell_16/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Т
"while/simple_rnn_cell_16/ones_likeFill1while/simple_rnn_cell_16/ones_like/Shape:output:01while/simple_rnn_cell_16/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2k
&while/simple_rnn_cell_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Л
$while/simple_rnn_cell_16/dropout/MulMul+while/simple_rnn_cell_16/ones_like:output:0/while/simple_rnn_cell_16/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
&while/simple_rnn_cell_16/dropout/ShapeShape+while/simple_rnn_cell_16/ones_like:output:0*
T0*
_output_shapes
:О
=while/simple_rnn_cell_16/dropout/random_uniform/RandomUniformRandomUniform/while/simple_rnn_cell_16/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype0t
/while/simple_rnn_cell_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ё
-while/simple_rnn_cell_16/dropout/GreaterEqualGreaterEqualFwhile/simple_rnn_cell_16/dropout/random_uniform/RandomUniform:output:08while/simple_rnn_cell_16/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ё
%while/simple_rnn_cell_16/dropout/CastCast1while/simple_rnn_cell_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2Д
&while/simple_rnn_cell_16/dropout/Mul_1Mul(while/simple_rnn_cell_16/dropout/Mul:z:0)while/simple_rnn_cell_16/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2m
*while/simple_rnn_cell_16/ones_like_1/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:o
*while/simple_rnn_cell_16/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Щ
$while/simple_rnn_cell_16/ones_like_1Fill3while/simple_rnn_cell_16/ones_like_1/Shape:output:03while/simple_rnn_cell_16/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџm
(while/simple_rnn_cell_16/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Т
&while/simple_rnn_cell_16/dropout_1/MulMul-while/simple_rnn_cell_16/ones_like_1:output:01while/simple_rnn_cell_16/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
(while/simple_rnn_cell_16/dropout_1/ShapeShape-while/simple_rnn_cell_16/ones_like_1:output:0*
T0*
_output_shapes
:У
?while/simple_rnn_cell_16/dropout_1/random_uniform/RandomUniformRandomUniform1while/simple_rnn_cell_16/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0v
1while/simple_rnn_cell_16/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ј
/while/simple_rnn_cell_16/dropout_1/GreaterEqualGreaterEqualHwhile/simple_rnn_cell_16/dropout_1/random_uniform/RandomUniform:output:0:while/simple_rnn_cell_16/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџІ
'while/simple_rnn_cell_16/dropout_1/CastCast3while/simple_rnn_cell_16/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЛ
(while/simple_rnn_cell_16/dropout_1/Mul_1Mul*while/simple_rnn_cell_16/dropout_1/Mul:z:0+while/simple_rnn_cell_16/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџГ
while/simple_rnn_cell_16/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/simple_rnn_cell_16/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2Љ
.while/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	2*
dtype0Ж
while/simple_rnn_cell_16/MatMulMatMul while/simple_rnn_cell_16/mul:z:06while/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЇ
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Т
 while/simple_rnn_cell_16/BiasAddBiasAdd)while/simple_rnn_cell_16/MatMul:product:07while/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/simple_rnn_cell_16/mul_1Mulwhile_placeholder_3,while/simple_rnn_cell_16/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЎ
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0М
!while/simple_rnn_cell_16/MatMul_1MatMul"while/simple_rnn_cell_16/mul_1:z:08while/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџА
while/simple_rnn_cell_16/addAddV2)while/simple_rnn_cell_16/BiasAdd:output:0+while/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџz
while/simple_rnn_cell_16/TanhTanh while/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџe
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
while/SelectV2SelectV2while/Tile:output:0!while/simple_rnn_cell_16/Tanh:y:0while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџg
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
while/SelectV2_1SelectV2while/Tile_1:output:0!while/simple_rnn_cell_16/Tanh:y:0while_placeholder_3*
T0*(
_output_shapes
:џџџџџџџџџР
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: u
while/Identity_4Identitywhile/SelectV2:output:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
while/Identity_5Identitywhile/SelectV2_1:output:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџт

while/NoOpNoOp0^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_simple_rnn_cell_16_biasadd_readvariableop_resource:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_16_matmul_readvariableop_resource9while_simple_rnn_cell_16_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"А
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : 2b
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_16/MatMul/ReadVariableOp.while/simple_rnn_cell_16/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
И
­
simple_rnn_17_while_cond_3573388
4simple_rnn_17_while_simple_rnn_17_while_loop_counter>
:simple_rnn_17_while_simple_rnn_17_while_maximum_iterations#
simple_rnn_17_while_placeholder%
!simple_rnn_17_while_placeholder_1%
!simple_rnn_17_while_placeholder_2%
!simple_rnn_17_while_placeholder_3:
6simple_rnn_17_while_less_simple_rnn_17_strided_slice_1P
Lsimple_rnn_17_while_simple_rnn_17_while_cond_357338___redundant_placeholder0P
Lsimple_rnn_17_while_simple_rnn_17_while_cond_357338___redundant_placeholder1P
Lsimple_rnn_17_while_simple_rnn_17_while_cond_357338___redundant_placeholder2P
Lsimple_rnn_17_while_simple_rnn_17_while_cond_357338___redundant_placeholder3P
Lsimple_rnn_17_while_simple_rnn_17_while_cond_357338___redundant_placeholder4 
simple_rnn_17_while_identity

simple_rnn_17/while/LessLesssimple_rnn_17_while_placeholder6simple_rnn_17_while_less_simple_rnn_17_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_17/while/IdentityIdentitysimple_rnn_17/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_17_while_identity%simple_rnn_17/while/Identity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
Ћ4
Є
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_355320

inputs,
simple_rnn_cell_16_355245:	2(
simple_rnn_cell_16_355247:	-
simple_rnn_cell_16_355249:

identityЂ*simple_rnn_cell_16/StatefulPartitionedCallЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maskю
*simple_rnn_cell_16/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_16_355245simple_rnn_cell_16_355247simple_rnn_cell_16_355249*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_355205n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_16_355245simple_rnn_cell_16_355247simple_rnn_cell_16_355249*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_355257*
condR
while_cond_355256*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ь
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ{
NoOpNoOp+^simple_rnn_cell_16/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ2: : : 2X
*simple_rnn_cell_16/StatefulPartitionedCall*simple_rnn_cell_16/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
 
_user_specified_nameinputs
н
с
-sequential_12_simple_rnn_17_while_cond_354900T
Psequential_12_simple_rnn_17_while_sequential_12_simple_rnn_17_while_loop_counterZ
Vsequential_12_simple_rnn_17_while_sequential_12_simple_rnn_17_while_maximum_iterations1
-sequential_12_simple_rnn_17_while_placeholder3
/sequential_12_simple_rnn_17_while_placeholder_13
/sequential_12_simple_rnn_17_while_placeholder_23
/sequential_12_simple_rnn_17_while_placeholder_3V
Rsequential_12_simple_rnn_17_while_less_sequential_12_simple_rnn_17_strided_slice_1l
hsequential_12_simple_rnn_17_while_sequential_12_simple_rnn_17_while_cond_354900___redundant_placeholder0l
hsequential_12_simple_rnn_17_while_sequential_12_simple_rnn_17_while_cond_354900___redundant_placeholder1l
hsequential_12_simple_rnn_17_while_sequential_12_simple_rnn_17_while_cond_354900___redundant_placeholder2l
hsequential_12_simple_rnn_17_while_sequential_12_simple_rnn_17_while_cond_354900___redundant_placeholder3l
hsequential_12_simple_rnn_17_while_sequential_12_simple_rnn_17_while_cond_354900___redundant_placeholder4.
*sequential_12_simple_rnn_17_while_identity
в
&sequential_12/simple_rnn_17/while/LessLess-sequential_12_simple_rnn_17_while_placeholderRsequential_12_simple_rnn_17_while_less_sequential_12_simple_rnn_17_strided_slice_1*
T0*
_output_shapes
: 
*sequential_12/simple_rnn_17/while/IdentityIdentity*sequential_12/simple_rnn_17/while/Less:z:0*
T0
*
_output_shapes
: "a
*sequential_12_simple_rnn_17_while_identity3sequential_12/simple_rnn_17/while/Identity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
Ь7
л
while_body_358229
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0M
9while_simple_rnn_cell_17_matmul_readvariableop_resource_0:
I
:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0:	O
;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorK
7while_simple_rnn_cell_17_matmul_readvariableop_resource:
G
8while_simple_rnn_cell_17_biasadd_readvariableop_resource:	M
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource:
Ђ/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpЂ.while/simple_rnn_cell_17/MatMul/ReadVariableOpЂ0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
(while/simple_rnn_cell_17/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(while/simple_rnn_cell_17/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?У
"while/simple_rnn_cell_17/ones_likeFill1while/simple_rnn_cell_17/ones_like/Shape:output:01while/simple_rnn_cell_17/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџm
*while/simple_rnn_cell_17/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:o
*while/simple_rnn_cell_17/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Щ
$while/simple_rnn_cell_17/ones_like_1Fill3while/simple_rnn_cell_17/ones_like_1/Shape:output:03while/simple_rnn_cell_17/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЕ
while/simple_rnn_cell_17/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/simple_rnn_cell_17/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЊ
.while/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_17_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ж
while/simple_rnn_cell_17/MatMulMatMul while/simple_rnn_cell_17/mul:z:06while/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЇ
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Т
 while/simple_rnn_cell_17/BiasAddBiasAdd)while/simple_rnn_cell_17/MatMul:product:07while/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/simple_rnn_cell_17/mul_1Mulwhile_placeholder_2-while/simple_rnn_cell_17/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЎ
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0М
!while/simple_rnn_cell_17/MatMul_1MatMul"while/simple_rnn_cell_17/mul_1:z:08while/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџА
while/simple_rnn_cell_17/addAddV2)while/simple_rnn_cell_17/BiasAdd:output:0+while/simple_rnn_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџz
while/simple_rnn_cell_17/TanhTanh while/simple_rnn_cell_17/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ђ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0!while/simple_rnn_cell_17/Tanh:y:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity!while/simple_rnn_cell_17/Tanh:y:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџт

while/NoOpNoOp0^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_17_biasadd_readvariableop_resource:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_17_matmul_readvariableop_resource9while_simple_rnn_cell_17_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџ: : : : : 2b
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_17/MatMul/ReadVariableOp.while/simple_rnn_cell_17/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Z
К
simple_rnn_17_while_body_3569758
4simple_rnn_17_while_simple_rnn_17_while_loop_counter>
:simple_rnn_17_while_simple_rnn_17_while_maximum_iterations#
simple_rnn_17_while_placeholder%
!simple_rnn_17_while_placeholder_1%
!simple_rnn_17_while_placeholder_2%
!simple_rnn_17_while_placeholder_37
3simple_rnn_17_while_simple_rnn_17_strided_slice_1_0s
osimple_rnn_17_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_tensorlistfromtensor_0w
ssimple_rnn_17_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_1_tensorlistfromtensor_0[
Gsimple_rnn_17_while_simple_rnn_cell_17_matmul_readvariableop_resource_0:
W
Hsimple_rnn_17_while_simple_rnn_cell_17_biasadd_readvariableop_resource_0:	]
Isimple_rnn_17_while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0:
 
simple_rnn_17_while_identity"
simple_rnn_17_while_identity_1"
simple_rnn_17_while_identity_2"
simple_rnn_17_while_identity_3"
simple_rnn_17_while_identity_4"
simple_rnn_17_while_identity_55
1simple_rnn_17_while_simple_rnn_17_strided_slice_1q
msimple_rnn_17_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_tensorlistfromtensoru
qsimple_rnn_17_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_1_tensorlistfromtensorY
Esimple_rnn_17_while_simple_rnn_cell_17_matmul_readvariableop_resource:
U
Fsimple_rnn_17_while_simple_rnn_cell_17_biasadd_readvariableop_resource:	[
Gsimple_rnn_17_while_simple_rnn_cell_17_matmul_1_readvariableop_resource:
Ђ=simple_rnn_17/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpЂ<simple_rnn_17/while/simple_rnn_cell_17/MatMul/ReadVariableOpЂ>simple_rnn_17/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp
Esimple_rnn_17/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   э
7simple_rnn_17/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_17_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_17_while_placeholderNsimple_rnn_17/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
Gsimple_rnn_17/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   є
9simple_rnn_17/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemssimple_rnn_17_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_1_tensorlistfromtensor_0simple_rnn_17_while_placeholderPsimple_rnn_17/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
Є
6simple_rnn_17/while/simple_rnn_cell_17/ones_like/ShapeShape>simple_rnn_17/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:{
6simple_rnn_17/while/simple_rnn_cell_17/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?э
0simple_rnn_17/while/simple_rnn_cell_17/ones_likeFill?simple_rnn_17/while/simple_rnn_cell_17/ones_like/Shape:output:0?simple_rnn_17/while/simple_rnn_cell_17/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
8simple_rnn_17/while/simple_rnn_cell_17/ones_like_1/ShapeShape!simple_rnn_17_while_placeholder_3*
T0*
_output_shapes
:}
8simple_rnn_17/while/simple_rnn_cell_17/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ѓ
2simple_rnn_17/while/simple_rnn_cell_17/ones_like_1FillAsimple_rnn_17/while/simple_rnn_cell_17/ones_like_1/Shape:output:0Asimple_rnn_17/while/simple_rnn_cell_17/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџп
*simple_rnn_17/while/simple_rnn_cell_17/mulMul>simple_rnn_17/while/TensorArrayV2Read/TensorListGetItem:item:09simple_rnn_17/while/simple_rnn_cell_17/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЦ
<simple_rnn_17/while/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_17_while_simple_rnn_cell_17_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0р
-simple_rnn_17/while/simple_rnn_cell_17/MatMulMatMul.simple_rnn_17/while/simple_rnn_cell_17/mul:z:0Dsimple_rnn_17/while/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџУ
=simple_rnn_17/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_17_while_simple_rnn_cell_17_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ь
.simple_rnn_17/while/simple_rnn_cell_17/BiasAddBiasAdd7simple_rnn_17/while/simple_rnn_cell_17/MatMul:product:0Esimple_rnn_17/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЦ
,simple_rnn_17/while/simple_rnn_cell_17/mul_1Mul!simple_rnn_17_while_placeholder_3;simple_rnn_17/while/simple_rnn_cell_17/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЪ
>simple_rnn_17/while/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_17_while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0ц
/simple_rnn_17/while/simple_rnn_cell_17/MatMul_1MatMul0simple_rnn_17/while/simple_rnn_cell_17/mul_1:z:0Fsimple_rnn_17/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџк
*simple_rnn_17/while/simple_rnn_cell_17/addAddV27simple_rnn_17/while/simple_rnn_cell_17/BiasAdd:output:09simple_rnn_17/while/simple_rnn_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
+simple_rnn_17/while/simple_rnn_cell_17/TanhTanh.simple_rnn_17/while/simple_rnn_cell_17/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџs
"simple_rnn_17/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      С
simple_rnn_17/while/TileTile@simple_rnn_17/while/TensorArrayV2Read_1/TensorListGetItem:item:0+simple_rnn_17/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџв
simple_rnn_17/while/SelectV2SelectV2!simple_rnn_17/while/Tile:output:0/simple_rnn_17/while/simple_rnn_cell_17/Tanh:y:0!simple_rnn_17_while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџu
$simple_rnn_17/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      Х
simple_rnn_17/while/Tile_1Tile@simple_rnn_17/while/TensorArrayV2Read_1/TensorListGetItem:item:0-simple_rnn_17/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџж
simple_rnn_17/while/SelectV2_1SelectV2#simple_rnn_17/while/Tile_1:output:0/simple_rnn_17/while/simple_rnn_cell_17/Tanh:y:0!simple_rnn_17_while_placeholder_3*
T0*(
_output_shapes
:џџџџџџџџџ
>simple_rnn_17/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B :  
8simple_rnn_17/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_17_while_placeholder_1Gsimple_rnn_17/while/TensorArrayV2Write/TensorListSetItem/index:output:0%simple_rnn_17/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшв[
simple_rnn_17/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_17/while/addAddV2simple_rnn_17_while_placeholder"simple_rnn_17/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_17/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_17/while/add_1AddV24simple_rnn_17_while_simple_rnn_17_while_loop_counter$simple_rnn_17/while/add_1/y:output:0*
T0*
_output_shapes
: 
simple_rnn_17/while/IdentityIdentitysimple_rnn_17/while/add_1:z:0^simple_rnn_17/while/NoOp*
T0*
_output_shapes
: Ђ
simple_rnn_17/while/Identity_1Identity:simple_rnn_17_while_simple_rnn_17_while_maximum_iterations^simple_rnn_17/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_17/while/Identity_2Identitysimple_rnn_17/while/add:z:0^simple_rnn_17/while/NoOp*
T0*
_output_shapes
: А
simple_rnn_17/while/Identity_3IdentityHsimple_rnn_17/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_17/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_17/while/Identity_4Identity%simple_rnn_17/while/SelectV2:output:0^simple_rnn_17/while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџЁ
simple_rnn_17/while/Identity_5Identity'simple_rnn_17/while/SelectV2_1:output:0^simple_rnn_17/while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_17/while/NoOpNoOp>^simple_rnn_17/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp=^simple_rnn_17/while/simple_rnn_cell_17/MatMul/ReadVariableOp?^simple_rnn_17/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_17_while_identity%simple_rnn_17/while/Identity:output:0"I
simple_rnn_17_while_identity_1'simple_rnn_17/while/Identity_1:output:0"I
simple_rnn_17_while_identity_2'simple_rnn_17/while/Identity_2:output:0"I
simple_rnn_17_while_identity_3'simple_rnn_17/while/Identity_3:output:0"I
simple_rnn_17_while_identity_4'simple_rnn_17/while/Identity_4:output:0"I
simple_rnn_17_while_identity_5'simple_rnn_17/while/Identity_5:output:0"h
1simple_rnn_17_while_simple_rnn_17_strided_slice_13simple_rnn_17_while_simple_rnn_17_strided_slice_1_0"
Fsimple_rnn_17_while_simple_rnn_cell_17_biasadd_readvariableop_resourceHsimple_rnn_17_while_simple_rnn_cell_17_biasadd_readvariableop_resource_0"
Gsimple_rnn_17_while_simple_rnn_cell_17_matmul_1_readvariableop_resourceIsimple_rnn_17_while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0"
Esimple_rnn_17_while_simple_rnn_cell_17_matmul_readvariableop_resourceGsimple_rnn_17_while_simple_rnn_cell_17_matmul_readvariableop_resource_0"ш
qsimple_rnn_17_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_1_tensorlistfromtensorssimple_rnn_17_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_1_tensorlistfromtensor_0"р
msimple_rnn_17_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_tensorlistfromtensorosimple_rnn_17_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_17_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : 2~
=simple_rnn_17/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp=simple_rnn_17/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp2|
<simple_rnn_17/while/simple_rnn_cell_17/MatMul/ReadVariableOp<simple_rnn_17/while/simple_rnn_cell_17/MatMul/ReadVariableOp2
>simple_rnn_17/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp>simple_rnn_17/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Й"
т
while_body_355399
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
!while_simple_rnn_cell_17_355421_0:
0
!while_simple_rnn_cell_17_355423_0:	5
!while_simple_rnn_cell_17_355425_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
while_simple_rnn_cell_17_355421:
.
while_simple_rnn_cell_17_355423:	3
while_simple_rnn_cell_17_355425:
Ђ0while/simple_rnn_cell_17/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0Љ
0while/simple_rnn_cell_17/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_17_355421_0!while_simple_rnn_cell_17_355423_0!while_simple_rnn_cell_17_355425_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_355385r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:09while/simple_rnn_cell_17/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity9while/simple_rnn_cell_17/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџ

while/NoOpNoOp1^while/simple_rnn_cell_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_17_355421!while_simple_rnn_cell_17_355421_0"D
while_simple_rnn_cell_17_355423!while_simple_rnn_cell_17_355423_0"D
while_simple_rnn_cell_17_355425!while_simple_rnn_cell_17_355425_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџ: : : : : 2d
0while/simple_rnn_cell_17/StatefulPartitionedCall0while/simple_rnn_cell_17/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
з


.__inference_sequential_12_layer_call_fn_356736

inputs
unknown:	'2
	unknown_0:	2
	unknown_1:	
	unknown_2:

	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	@
	unknown_7:@
	unknown_8:@
	unknown_9:
identityЂStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_356029o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ю
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_355385

inputs

states2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	4
 matmul_1_readvariableop_resource:

identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџG
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
mulMulinputsones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0k
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ]
mul_1Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
MatMul_1MatMul	mul_1:z:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџH
TanhTanhadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџX
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџZ

Identity_1IdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:PL
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates
 
Я
I__inference_sequential_12_layer_call_and_return_conditional_losses_356674
embedding_12_input&
embedding_12_356644:	'2'
simple_rnn_16_356649:	2#
simple_rnn_16_356651:	(
simple_rnn_16_356653:
(
simple_rnn_17_356656:
#
simple_rnn_17_356658:	(
simple_rnn_17_356660:
"
dense_24_356663:	@
dense_24_356665:@!
dense_25_356668:@
dense_25_356670:
identityЂ dense_24/StatefulPartitionedCallЂ dense_25/StatefulPartitionedCallЂ$embedding_12/StatefulPartitionedCallЂ%simple_rnn_16/StatefulPartitionedCallЂ%simple_rnn_17/StatefulPartitionedCallљ
$embedding_12/StatefulPartitionedCallStatefulPartitionedCallembedding_12_inputembedding_12_356644*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_12_layer_call_and_return_conditional_losses_355674\
embedding_12/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
embedding_12/NotEqualNotEqualembedding_12_input embedding_12/NotEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџф
%simple_rnn_16/StatefulPartitionedCallStatefulPartitionedCall-embedding_12/StatefulPartitionedCall:output:0embedding_12/NotEqual:z:0simple_rnn_16_356649simple_rnn_16_356651simple_rnn_16_356653*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_356477с
%simple_rnn_17/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_16/StatefulPartitionedCall:output:0embedding_12/NotEqual:z:0simple_rnn_17_356656simple_rnn_17_356658simple_rnn_17_356660*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_356272
 dense_24/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_17/StatefulPartitionedCall:output:0dense_24_356663dense_24_356665*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_356005
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_356668dense_25_356670*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_356022x
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall%^embedding_12/StatefulPartitionedCall&^simple_rnn_16/StatefulPartitionedCall&^simple_rnn_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall2N
%simple_rnn_16/StatefulPartitionedCall%simple_rnn_16/StatefulPartitionedCall2N
%simple_rnn_17/StatefulPartitionedCall%simple_rnn_17/StatefulPartitionedCall:[ W
'
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameembedding_12_input
	
љ
while_cond_355738
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_355738___redundant_placeholder04
0while_while_cond_355738___redundant_placeholder14
0while_while_cond_355738___redundant_placeholder24
0while_while_cond_355738___redundant_placeholder34
0while_while_cond_355738___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
Ш

р
3__inference_simple_rnn_cell_17_layer_call_fn_358944

inputs
states_0
unknown:

	unknown_0:	
	unknown_1:

identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_355385p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
с
У
I__inference_sequential_12_layer_call_and_return_conditional_losses_356029

inputs&
embedding_12_355675:	'2'
simple_rnn_16_355829:	2#
simple_rnn_16_355831:	(
simple_rnn_16_355833:
(
simple_rnn_17_355987:
#
simple_rnn_17_355989:	(
simple_rnn_17_355991:
"
dense_24_356006:	@
dense_24_356008:@!
dense_25_356023:@
dense_25_356025:
identityЂ dense_24/StatefulPartitionedCallЂ dense_25/StatefulPartitionedCallЂ$embedding_12/StatefulPartitionedCallЂ%simple_rnn_16/StatefulPartitionedCallЂ%simple_rnn_17/StatefulPartitionedCallэ
$embedding_12/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_12_355675*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_12_layer_call_and_return_conditional_losses_355674\
embedding_12/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
embedding_12/NotEqualNotEqualinputs embedding_12/NotEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџф
%simple_rnn_16/StatefulPartitionedCallStatefulPartitionedCall-embedding_12/StatefulPartitionedCall:output:0embedding_12/NotEqual:z:0simple_rnn_16_355829simple_rnn_16_355831simple_rnn_16_355833*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_355828с
%simple_rnn_17/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_16/StatefulPartitionedCall:output:0embedding_12/NotEqual:z:0simple_rnn_17_355987simple_rnn_17_355989simple_rnn_17_355991*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_355986
 dense_24/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_17/StatefulPartitionedCall:output:0dense_24_356006dense_24_356008*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_356005
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_356023dense_25_356025*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_356022x
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall%^embedding_12/StatefulPartitionedCall&^simple_rnn_16/StatefulPartitionedCall&^simple_rnn_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall2N
%simple_rnn_16/StatefulPartitionedCall%simple_rnn_16/StatefulPartitionedCall2N
%simple_rnn_17/StatefulPartitionedCall%simple_rnn_17/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
!
р
while_body_355257
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
!while_simple_rnn_cell_16_355279_0:	20
!while_simple_rnn_cell_16_355281_0:	5
!while_simple_rnn_cell_16_355283_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
while_simple_rnn_cell_16_355279:	2.
while_simple_rnn_cell_16_355281:	3
while_simple_rnn_cell_16_355283:
Ђ0while/simple_rnn_cell_16/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ2*
element_dtype0Љ
0while/simple_rnn_cell_16/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_16_355279_0!while_simple_rnn_cell_16_355281_0!while_simple_rnn_cell_16_355283_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_355205т
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_16/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity9while/simple_rnn_cell_16/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџ

while/NoOpNoOp1^while/simple_rnn_cell_16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_16_355279!while_simple_rnn_cell_16_355279_0"D
while_simple_rnn_cell_16_355281!while_simple_rnn_cell_16_355281_0"D
while_simple_rnn_cell_16_355283!while_simple_rnn_cell_16_355283_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџ: : : : : 2d
0while/simple_rnn_cell_16/StatefulPartitionedCall0while/simple_rnn_cell_16/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
є­
Ю
"__inference__traced_restore_359309
file_prefix;
(assignvariableop_embedding_12_embeddings:	'25
"assignvariableop_1_dense_24_kernel:	@.
 assignvariableop_2_dense_24_bias:@4
"assignvariableop_3_dense_25_kernel:@.
 assignvariableop_4_dense_25_bias:M
:assignvariableop_5_simple_rnn_16_simple_rnn_cell_16_kernel:	2X
Dassignvariableop_6_simple_rnn_16_simple_rnn_cell_16_recurrent_kernel:
G
8assignvariableop_7_simple_rnn_16_simple_rnn_cell_16_bias:	N
:assignvariableop_8_simple_rnn_17_simple_rnn_cell_17_kernel:
X
Dassignvariableop_9_simple_rnn_17_simple_rnn_cell_17_recurrent_kernel:
H
9assignvariableop_10_simple_rnn_17_simple_rnn_cell_17_bias:	'
assignvariableop_11_adam_iter:	 )
assignvariableop_12_adam_beta_1: )
assignvariableop_13_adam_beta_2: (
assignvariableop_14_adam_decay: 0
&assignvariableop_15_adam_learning_rate: %
assignvariableop_16_total_1: %
assignvariableop_17_count_1: #
assignvariableop_18_total: #
assignvariableop_19_count: E
2assignvariableop_20_adam_embedding_12_embeddings_m:	'2=
*assignvariableop_21_adam_dense_24_kernel_m:	@6
(assignvariableop_22_adam_dense_24_bias_m:@<
*assignvariableop_23_adam_dense_25_kernel_m:@6
(assignvariableop_24_adam_dense_25_bias_m:U
Bassignvariableop_25_adam_simple_rnn_16_simple_rnn_cell_16_kernel_m:	2`
Lassignvariableop_26_adam_simple_rnn_16_simple_rnn_cell_16_recurrent_kernel_m:
O
@assignvariableop_27_adam_simple_rnn_16_simple_rnn_cell_16_bias_m:	V
Bassignvariableop_28_adam_simple_rnn_17_simple_rnn_cell_17_kernel_m:
`
Lassignvariableop_29_adam_simple_rnn_17_simple_rnn_cell_17_recurrent_kernel_m:
O
@assignvariableop_30_adam_simple_rnn_17_simple_rnn_cell_17_bias_m:	E
2assignvariableop_31_adam_embedding_12_embeddings_v:	'2=
*assignvariableop_32_adam_dense_24_kernel_v:	@6
(assignvariableop_33_adam_dense_24_bias_v:@<
*assignvariableop_34_adam_dense_25_kernel_v:@6
(assignvariableop_35_adam_dense_25_bias_v:U
Bassignvariableop_36_adam_simple_rnn_16_simple_rnn_cell_16_kernel_v:	2`
Lassignvariableop_37_adam_simple_rnn_16_simple_rnn_cell_16_recurrent_kernel_v:
O
@assignvariableop_38_adam_simple_rnn_16_simple_rnn_cell_16_bias_v:	V
Bassignvariableop_39_adam_simple_rnn_17_simple_rnn_cell_17_kernel_v:
`
Lassignvariableop_40_adam_simple_rnn_17_simple_rnn_cell_17_recurrent_kernel_v:
O
@assignvariableop_41_adam_simple_rnn_17_simple_rnn_cell_17_bias_v:	
identity_43ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ъ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*№
valueцBу+B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЦ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ј
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Т
_output_shapesЏ
Ќ:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp(assignvariableop_embedding_12_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_24_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense_24_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_25_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_25_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_5AssignVariableOp:assignvariableop_5_simple_rnn_16_simple_rnn_cell_16_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOp_6AssignVariableOpDassignvariableop_6_simple_rnn_16_simple_rnn_cell_16_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_7AssignVariableOp8assignvariableop_7_simple_rnn_16_simple_rnn_cell_16_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_8AssignVariableOp:assignvariableop_8_simple_rnn_17_simple_rnn_cell_17_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOp_9AssignVariableOpDassignvariableop_9_simple_rnn_17_simple_rnn_cell_17_recurrent_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_10AssignVariableOp9assignvariableop_10_simple_rnn_17_simple_rnn_cell_17_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_iterIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_decayIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp&assignvariableop_15_adam_learning_rateIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_20AssignVariableOp2assignvariableop_20_adam_embedding_12_embeddings_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_24_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_24_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_25_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_25_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOp_25AssignVariableOpBassignvariableop_25_adam_simple_rnn_16_simple_rnn_cell_16_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_26AssignVariableOpLassignvariableop_26_adam_simple_rnn_16_simple_rnn_cell_16_recurrent_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_simple_rnn_16_simple_rnn_cell_16_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOp_28AssignVariableOpBassignvariableop_28_adam_simple_rnn_17_simple_rnn_cell_17_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_29AssignVariableOpLassignvariableop_29_adam_simple_rnn_17_simple_rnn_cell_17_recurrent_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_30AssignVariableOp@assignvariableop_30_adam_simple_rnn_17_simple_rnn_cell_17_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_31AssignVariableOp2assignvariableop_31_adam_embedding_12_embeddings_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_dense_24_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_dense_24_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_dense_25_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_dense_25_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOp_36AssignVariableOpBassignvariableop_36_adam_simple_rnn_16_simple_rnn_cell_16_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_37AssignVariableOpLassignvariableop_37_adam_simple_rnn_16_simple_rnn_cell_16_recurrent_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_38AssignVariableOp@assignvariableop_38_adam_simple_rnn_16_simple_rnn_cell_16_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOp_39AssignVariableOpBassignvariableop_39_adam_simple_rnn_17_simple_rnn_cell_17_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_40AssignVariableOpLassignvariableop_40_adam_simple_rnn_17_simple_rnn_cell_17_recurrent_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_41AssignVariableOp@assignvariableop_41_adam_simple_rnn_17_simple_rnn_cell_17_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ы
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_43IdentityIdentity_42:output:0^NoOp_1*
T0*
_output_shapes
: и
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_43Identity_43:output:0*i
_input_shapesX
V: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Љm
К
-sequential_12_simple_rnn_17_while_body_354901T
Psequential_12_simple_rnn_17_while_sequential_12_simple_rnn_17_while_loop_counterZ
Vsequential_12_simple_rnn_17_while_sequential_12_simple_rnn_17_while_maximum_iterations1
-sequential_12_simple_rnn_17_while_placeholder3
/sequential_12_simple_rnn_17_while_placeholder_13
/sequential_12_simple_rnn_17_while_placeholder_23
/sequential_12_simple_rnn_17_while_placeholder_3S
Osequential_12_simple_rnn_17_while_sequential_12_simple_rnn_17_strided_slice_1_0
sequential_12_simple_rnn_17_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_17_tensorarrayunstack_tensorlistfromtensor_0
sequential_12_simple_rnn_17_while_tensorarrayv2read_1_tensorlistgetitem_sequential_12_simple_rnn_17_tensorarrayunstack_1_tensorlistfromtensor_0i
Usequential_12_simple_rnn_17_while_simple_rnn_cell_17_matmul_readvariableop_resource_0:
e
Vsequential_12_simple_rnn_17_while_simple_rnn_cell_17_biasadd_readvariableop_resource_0:	k
Wsequential_12_simple_rnn_17_while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0:
.
*sequential_12_simple_rnn_17_while_identity0
,sequential_12_simple_rnn_17_while_identity_10
,sequential_12_simple_rnn_17_while_identity_20
,sequential_12_simple_rnn_17_while_identity_30
,sequential_12_simple_rnn_17_while_identity_40
,sequential_12_simple_rnn_17_while_identity_5Q
Msequential_12_simple_rnn_17_while_sequential_12_simple_rnn_17_strided_slice_1
sequential_12_simple_rnn_17_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_17_tensorarrayunstack_tensorlistfromtensor
sequential_12_simple_rnn_17_while_tensorarrayv2read_1_tensorlistgetitem_sequential_12_simple_rnn_17_tensorarrayunstack_1_tensorlistfromtensorg
Ssequential_12_simple_rnn_17_while_simple_rnn_cell_17_matmul_readvariableop_resource:
c
Tsequential_12_simple_rnn_17_while_simple_rnn_cell_17_biasadd_readvariableop_resource:	i
Usequential_12_simple_rnn_17_while_simple_rnn_cell_17_matmul_1_readvariableop_resource:
ЂKsequential_12/simple_rnn_17/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpЂJsequential_12/simple_rnn_17/while/simple_rnn_cell_17/MatMul/ReadVariableOpЂLsequential_12/simple_rnn_17/while/simple_rnn_cell_17/MatMul_1/ReadVariableOpЄ
Ssequential_12/simple_rnn_17/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Д
Esequential_12/simple_rnn_17/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_12_simple_rnn_17_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_17_tensorarrayunstack_tensorlistfromtensor_0-sequential_12_simple_rnn_17_while_placeholder\sequential_12/simple_rnn_17/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0І
Usequential_12/simple_rnn_17/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Л
Gsequential_12/simple_rnn_17/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemsequential_12_simple_rnn_17_while_tensorarrayv2read_1_tensorlistgetitem_sequential_12_simple_rnn_17_tensorarrayunstack_1_tensorlistfromtensor_0-sequential_12_simple_rnn_17_while_placeholder^sequential_12/simple_rnn_17/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
Р
Dsequential_12/simple_rnn_17/while/simple_rnn_cell_17/ones_like/ShapeShapeLsequential_12/simple_rnn_17/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:
Dsequential_12/simple_rnn_17/while/simple_rnn_cell_17/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
>sequential_12/simple_rnn_17/while/simple_rnn_cell_17/ones_likeFillMsequential_12/simple_rnn_17/while/simple_rnn_cell_17/ones_like/Shape:output:0Msequential_12/simple_rnn_17/while/simple_rnn_cell_17/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЅ
Fsequential_12/simple_rnn_17/while/simple_rnn_cell_17/ones_like_1/ShapeShape/sequential_12_simple_rnn_17_while_placeholder_3*
T0*
_output_shapes
:
Fsequential_12/simple_rnn_17/while/simple_rnn_cell_17/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
@sequential_12/simple_rnn_17/while/simple_rnn_cell_17/ones_like_1FillOsequential_12/simple_rnn_17/while/simple_rnn_cell_17/ones_like_1/Shape:output:0Osequential_12/simple_rnn_17/while/simple_rnn_cell_17/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
8sequential_12/simple_rnn_17/while/simple_rnn_cell_17/mulMulLsequential_12/simple_rnn_17/while/TensorArrayV2Read/TensorListGetItem:item:0Gsequential_12/simple_rnn_17/while/simple_rnn_cell_17/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџт
Jsequential_12/simple_rnn_17/while/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOpUsequential_12_simple_rnn_17_while_simple_rnn_cell_17_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
;sequential_12/simple_rnn_17/while/simple_rnn_cell_17/MatMulMatMul<sequential_12/simple_rnn_17/while/simple_rnn_cell_17/mul:z:0Rsequential_12/simple_rnn_17/while/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџп
Ksequential_12/simple_rnn_17/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOpVsequential_12_simple_rnn_17_while_simple_rnn_cell_17_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0
<sequential_12/simple_rnn_17/while/simple_rnn_cell_17/BiasAddBiasAddEsequential_12/simple_rnn_17/while/simple_rnn_cell_17/MatMul:product:0Ssequential_12/simple_rnn_17/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ№
:sequential_12/simple_rnn_17/while/simple_rnn_cell_17/mul_1Mul/sequential_12_simple_rnn_17_while_placeholder_3Isequential_12/simple_rnn_17/while/simple_rnn_cell_17/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџц
Lsequential_12/simple_rnn_17/while/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOpWsequential_12_simple_rnn_17_while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
=sequential_12/simple_rnn_17/while/simple_rnn_cell_17/MatMul_1MatMul>sequential_12/simple_rnn_17/while/simple_rnn_cell_17/mul_1:z:0Tsequential_12/simple_rnn_17/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
8sequential_12/simple_rnn_17/while/simple_rnn_cell_17/addAddV2Esequential_12/simple_rnn_17/while/simple_rnn_cell_17/BiasAdd:output:0Gsequential_12/simple_rnn_17/while/simple_rnn_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџВ
9sequential_12/simple_rnn_17/while/simple_rnn_cell_17/TanhTanh<sequential_12/simple_rnn_17/while/simple_rnn_cell_17/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
0sequential_12/simple_rnn_17/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ы
&sequential_12/simple_rnn_17/while/TileTileNsequential_12/simple_rnn_17/while/TensorArrayV2Read_1/TensorListGetItem:item:09sequential_12/simple_rnn_17/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
*sequential_12/simple_rnn_17/while/SelectV2SelectV2/sequential_12/simple_rnn_17/while/Tile:output:0=sequential_12/simple_rnn_17/while/simple_rnn_cell_17/Tanh:y:0/sequential_12_simple_rnn_17_while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџ
2sequential_12/simple_rnn_17/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      я
(sequential_12/simple_rnn_17/while/Tile_1TileNsequential_12/simple_rnn_17/while/TensorArrayV2Read_1/TensorListGetItem:item:0;sequential_12/simple_rnn_17/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
,sequential_12/simple_rnn_17/while/SelectV2_1SelectV21sequential_12/simple_rnn_17/while/Tile_1:output:0=sequential_12/simple_rnn_17/while/simple_rnn_cell_17/Tanh:y:0/sequential_12_simple_rnn_17_while_placeholder_3*
T0*(
_output_shapes
:џџџџџџџџџ
Lsequential_12/simple_rnn_17/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : и
Fsequential_12/simple_rnn_17/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem/sequential_12_simple_rnn_17_while_placeholder_1Usequential_12/simple_rnn_17/while/TensorArrayV2Write/TensorListSetItem/index:output:03sequential_12/simple_rnn_17/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшвi
'sequential_12/simple_rnn_17/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :А
%sequential_12/simple_rnn_17/while/addAddV2-sequential_12_simple_rnn_17_while_placeholder0sequential_12/simple_rnn_17/while/add/y:output:0*
T0*
_output_shapes
: k
)sequential_12/simple_rnn_17/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :з
'sequential_12/simple_rnn_17/while/add_1AddV2Psequential_12_simple_rnn_17_while_sequential_12_simple_rnn_17_while_loop_counter2sequential_12/simple_rnn_17/while/add_1/y:output:0*
T0*
_output_shapes
: ­
*sequential_12/simple_rnn_17/while/IdentityIdentity+sequential_12/simple_rnn_17/while/add_1:z:0'^sequential_12/simple_rnn_17/while/NoOp*
T0*
_output_shapes
: к
,sequential_12/simple_rnn_17/while/Identity_1IdentityVsequential_12_simple_rnn_17_while_sequential_12_simple_rnn_17_while_maximum_iterations'^sequential_12/simple_rnn_17/while/NoOp*
T0*
_output_shapes
: ­
,sequential_12/simple_rnn_17/while/Identity_2Identity)sequential_12/simple_rnn_17/while/add:z:0'^sequential_12/simple_rnn_17/while/NoOp*
T0*
_output_shapes
: к
,sequential_12/simple_rnn_17/while/Identity_3IdentityVsequential_12/simple_rnn_17/while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^sequential_12/simple_rnn_17/while/NoOp*
T0*
_output_shapes
: Щ
,sequential_12/simple_rnn_17/while/Identity_4Identity3sequential_12/simple_rnn_17/while/SelectV2:output:0'^sequential_12/simple_rnn_17/while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџЫ
,sequential_12/simple_rnn_17/while/Identity_5Identity5sequential_12/simple_rnn_17/while/SelectV2_1:output:0'^sequential_12/simple_rnn_17/while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџв
&sequential_12/simple_rnn_17/while/NoOpNoOpL^sequential_12/simple_rnn_17/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpK^sequential_12/simple_rnn_17/while/simple_rnn_cell_17/MatMul/ReadVariableOpM^sequential_12/simple_rnn_17/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "a
*sequential_12_simple_rnn_17_while_identity3sequential_12/simple_rnn_17/while/Identity:output:0"e
,sequential_12_simple_rnn_17_while_identity_15sequential_12/simple_rnn_17/while/Identity_1:output:0"e
,sequential_12_simple_rnn_17_while_identity_25sequential_12/simple_rnn_17/while/Identity_2:output:0"e
,sequential_12_simple_rnn_17_while_identity_35sequential_12/simple_rnn_17/while/Identity_3:output:0"e
,sequential_12_simple_rnn_17_while_identity_45sequential_12/simple_rnn_17/while/Identity_4:output:0"e
,sequential_12_simple_rnn_17_while_identity_55sequential_12/simple_rnn_17/while/Identity_5:output:0" 
Msequential_12_simple_rnn_17_while_sequential_12_simple_rnn_17_strided_slice_1Osequential_12_simple_rnn_17_while_sequential_12_simple_rnn_17_strided_slice_1_0"Ў
Tsequential_12_simple_rnn_17_while_simple_rnn_cell_17_biasadd_readvariableop_resourceVsequential_12_simple_rnn_17_while_simple_rnn_cell_17_biasadd_readvariableop_resource_0"А
Usequential_12_simple_rnn_17_while_simple_rnn_cell_17_matmul_1_readvariableop_resourceWsequential_12_simple_rnn_17_while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0"Ќ
Ssequential_12_simple_rnn_17_while_simple_rnn_cell_17_matmul_readvariableop_resourceUsequential_12_simple_rnn_17_while_simple_rnn_cell_17_matmul_readvariableop_resource_0"Ђ
sequential_12_simple_rnn_17_while_tensorarrayv2read_1_tensorlistgetitem_sequential_12_simple_rnn_17_tensorarrayunstack_1_tensorlistfromtensorsequential_12_simple_rnn_17_while_tensorarrayv2read_1_tensorlistgetitem_sequential_12_simple_rnn_17_tensorarrayunstack_1_tensorlistfromtensor_0"
sequential_12_simple_rnn_17_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_17_tensorarrayunstack_tensorlistfromtensorsequential_12_simple_rnn_17_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_17_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : 2
Ksequential_12/simple_rnn_17/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpKsequential_12/simple_rnn_17/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp2
Jsequential_12/simple_rnn_17/while/simple_rnn_cell_17/MatMul/ReadVariableOpJsequential_12/simple_rnn_17/while/simple_rnn_cell_17/MatMul/ReadVariableOp2
Lsequential_12/simple_rnn_17/while/simple_rnn_cell_17/MatMul_1/ReadVariableOpLsequential_12/simple_rnn_17/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ћ4
Є
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_355137

inputs,
simple_rnn_cell_16_355062:	2(
simple_rnn_cell_16_355064:	-
simple_rnn_cell_16_355066:

identityЂ*simple_rnn_cell_16/StatefulPartitionedCallЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maskю
*simple_rnn_cell_16/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_16_355062simple_rnn_cell_16_355064simple_rnn_cell_16_355066*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_355061n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_16_355062simple_rnn_cell_16_355064simple_rnn_cell_16_355066*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_355074*
condR
while_cond_355073*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ь
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ{
NoOpNoOp+^simple_rnn_cell_16/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ2: : : 2X
*simple_rnn_cell_16/StatefulPartitionedCall*simple_rnn_cell_16/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
 
_user_specified_nameinputs
м
Њ
while_cond_355256
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_355256___redundant_placeholder04
0while_while_cond_355256___redundant_placeholder14
0while_while_cond_355256___redundant_placeholder24
0while_while_cond_355256___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Ш

р
3__inference_simple_rnn_cell_17_layer_call_fn_358958

inputs
states_0
unknown:

	unknown_0:	
	unknown_1:

identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_355531p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
 
Я
I__inference_sequential_12_layer_call_and_return_conditional_losses_356641
embedding_12_input&
embedding_12_356611:	'2'
simple_rnn_16_356616:	2#
simple_rnn_16_356618:	(
simple_rnn_16_356620:
(
simple_rnn_17_356623:
#
simple_rnn_17_356625:	(
simple_rnn_17_356627:
"
dense_24_356630:	@
dense_24_356632:@!
dense_25_356635:@
dense_25_356637:
identityЂ dense_24/StatefulPartitionedCallЂ dense_25/StatefulPartitionedCallЂ$embedding_12/StatefulPartitionedCallЂ%simple_rnn_16/StatefulPartitionedCallЂ%simple_rnn_17/StatefulPartitionedCallљ
$embedding_12/StatefulPartitionedCallStatefulPartitionedCallembedding_12_inputembedding_12_356611*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_12_layer_call_and_return_conditional_losses_355674\
embedding_12/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
embedding_12/NotEqualNotEqualembedding_12_input embedding_12/NotEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџф
%simple_rnn_16/StatefulPartitionedCallStatefulPartitionedCall-embedding_12/StatefulPartitionedCall:output:0embedding_12/NotEqual:z:0simple_rnn_16_356616simple_rnn_16_356618simple_rnn_16_356620*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_355828с
%simple_rnn_17/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_16/StatefulPartitionedCall:output:0embedding_12/NotEqual:z:0simple_rnn_17_356623simple_rnn_17_356625simple_rnn_17_356627*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_355986
 dense_24/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_17/StatefulPartitionedCall:output:0dense_24_356630dense_24_356632*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_356005
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_356635dense_25_356637*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_356022x
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall%^embedding_12/StatefulPartitionedCall&^simple_rnn_16/StatefulPartitionedCall&^simple_rnn_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall2N
%simple_rnn_16/StatefulPartitionedCall%simple_rnn_16/StatefulPartitionedCall2N
%simple_rnn_17/StatefulPartitionedCall%simple_rnn_17/StatefulPartitionedCall:[ W
'
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameembedding_12_input
ћ

Ћ
.__inference_sequential_12_layer_call_fn_356608
embedding_12_input
unknown:	'2
	unknown_0:	2
	unknown_1:	
	unknown_2:

	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	@
	unknown_7:@
	unknown_8:@
	unknown_9:
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallembedding_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_356556o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameembedding_12_input
бX
И
simple_rnn_16_while_body_3568308
4simple_rnn_16_while_simple_rnn_16_while_loop_counter>
:simple_rnn_16_while_simple_rnn_16_while_maximum_iterations#
simple_rnn_16_while_placeholder%
!simple_rnn_16_while_placeholder_1%
!simple_rnn_16_while_placeholder_2%
!simple_rnn_16_while_placeholder_37
3simple_rnn_16_while_simple_rnn_16_strided_slice_1_0s
osimple_rnn_16_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_tensorlistfromtensor_0w
ssimple_rnn_16_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_1_tensorlistfromtensor_0Z
Gsimple_rnn_16_while_simple_rnn_cell_16_matmul_readvariableop_resource_0:	2W
Hsimple_rnn_16_while_simple_rnn_cell_16_biasadd_readvariableop_resource_0:	]
Isimple_rnn_16_while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0:
 
simple_rnn_16_while_identity"
simple_rnn_16_while_identity_1"
simple_rnn_16_while_identity_2"
simple_rnn_16_while_identity_3"
simple_rnn_16_while_identity_4"
simple_rnn_16_while_identity_55
1simple_rnn_16_while_simple_rnn_16_strided_slice_1q
msimple_rnn_16_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_tensorlistfromtensoru
qsimple_rnn_16_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_1_tensorlistfromtensorX
Esimple_rnn_16_while_simple_rnn_cell_16_matmul_readvariableop_resource:	2U
Fsimple_rnn_16_while_simple_rnn_cell_16_biasadd_readvariableop_resource:	[
Gsimple_rnn_16_while_simple_rnn_cell_16_matmul_1_readvariableop_resource:
Ђ=simple_rnn_16/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpЂ<simple_rnn_16/while/simple_rnn_cell_16/MatMul/ReadVariableOpЂ>simple_rnn_16/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp
Esimple_rnn_16/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   ь
7simple_rnn_16/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_16_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_16_while_placeholderNsimple_rnn_16/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ2*
element_dtype0
Gsimple_rnn_16/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   є
9simple_rnn_16/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemssimple_rnn_16_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_1_tensorlistfromtensor_0simple_rnn_16_while_placeholderPsimple_rnn_16/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
Є
6simple_rnn_16/while/simple_rnn_cell_16/ones_like/ShapeShape>simple_rnn_16/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:{
6simple_rnn_16/while/simple_rnn_cell_16/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ь
0simple_rnn_16/while/simple_rnn_cell_16/ones_likeFill?simple_rnn_16/while/simple_rnn_cell_16/ones_like/Shape:output:0?simple_rnn_16/while/simple_rnn_cell_16/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
8simple_rnn_16/while/simple_rnn_cell_16/ones_like_1/ShapeShape!simple_rnn_16_while_placeholder_3*
T0*
_output_shapes
:}
8simple_rnn_16/while/simple_rnn_cell_16/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ѓ
2simple_rnn_16/while/simple_rnn_cell_16/ones_like_1FillAsimple_rnn_16/while/simple_rnn_cell_16/ones_like_1/Shape:output:0Asimple_rnn_16/while/simple_rnn_cell_16/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџо
*simple_rnn_16/while/simple_rnn_cell_16/mulMul>simple_rnn_16/while/TensorArrayV2Read/TensorListGetItem:item:09simple_rnn_16/while/simple_rnn_cell_16/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2Х
<simple_rnn_16/while/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_16_while_simple_rnn_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	2*
dtype0р
-simple_rnn_16/while/simple_rnn_cell_16/MatMulMatMul.simple_rnn_16/while/simple_rnn_cell_16/mul:z:0Dsimple_rnn_16/while/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџУ
=simple_rnn_16/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_16_while_simple_rnn_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ь
.simple_rnn_16/while/simple_rnn_cell_16/BiasAddBiasAdd7simple_rnn_16/while/simple_rnn_cell_16/MatMul:product:0Esimple_rnn_16/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЦ
,simple_rnn_16/while/simple_rnn_cell_16/mul_1Mul!simple_rnn_16_while_placeholder_3;simple_rnn_16/while/simple_rnn_cell_16/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЪ
>simple_rnn_16/while/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_16_while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0ц
/simple_rnn_16/while/simple_rnn_cell_16/MatMul_1MatMul0simple_rnn_16/while/simple_rnn_cell_16/mul_1:z:0Fsimple_rnn_16/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџк
*simple_rnn_16/while/simple_rnn_cell_16/addAddV27simple_rnn_16/while/simple_rnn_cell_16/BiasAdd:output:09simple_rnn_16/while/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
+simple_rnn_16/while/simple_rnn_cell_16/TanhTanh.simple_rnn_16/while/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџs
"simple_rnn_16/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      С
simple_rnn_16/while/TileTile@simple_rnn_16/while/TensorArrayV2Read_1/TensorListGetItem:item:0+simple_rnn_16/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџв
simple_rnn_16/while/SelectV2SelectV2!simple_rnn_16/while/Tile:output:0/simple_rnn_16/while/simple_rnn_cell_16/Tanh:y:0!simple_rnn_16_while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџu
$simple_rnn_16/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      Х
simple_rnn_16/while/Tile_1Tile@simple_rnn_16/while/TensorArrayV2Read_1/TensorListGetItem:item:0-simple_rnn_16/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџж
simple_rnn_16/while/SelectV2_1SelectV2#simple_rnn_16/while/Tile_1:output:0/simple_rnn_16/while/simple_rnn_cell_16/Tanh:y:0!simple_rnn_16_while_placeholder_3*
T0*(
_output_shapes
:џџџџџџџџџј
8simple_rnn_16/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_16_while_placeholder_1simple_rnn_16_while_placeholder%simple_rnn_16/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшв[
simple_rnn_16/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_16/while/addAddV2simple_rnn_16_while_placeholder"simple_rnn_16/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_16/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_16/while/add_1AddV24simple_rnn_16_while_simple_rnn_16_while_loop_counter$simple_rnn_16/while/add_1/y:output:0*
T0*
_output_shapes
: 
simple_rnn_16/while/IdentityIdentitysimple_rnn_16/while/add_1:z:0^simple_rnn_16/while/NoOp*
T0*
_output_shapes
: Ђ
simple_rnn_16/while/Identity_1Identity:simple_rnn_16_while_simple_rnn_16_while_maximum_iterations^simple_rnn_16/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_16/while/Identity_2Identitysimple_rnn_16/while/add:z:0^simple_rnn_16/while/NoOp*
T0*
_output_shapes
: А
simple_rnn_16/while/Identity_3IdentityHsimple_rnn_16/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_16/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_16/while/Identity_4Identity%simple_rnn_16/while/SelectV2:output:0^simple_rnn_16/while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџЁ
simple_rnn_16/while/Identity_5Identity'simple_rnn_16/while/SelectV2_1:output:0^simple_rnn_16/while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_16/while/NoOpNoOp>^simple_rnn_16/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp=^simple_rnn_16/while/simple_rnn_cell_16/MatMul/ReadVariableOp?^simple_rnn_16/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_16_while_identity%simple_rnn_16/while/Identity:output:0"I
simple_rnn_16_while_identity_1'simple_rnn_16/while/Identity_1:output:0"I
simple_rnn_16_while_identity_2'simple_rnn_16/while/Identity_2:output:0"I
simple_rnn_16_while_identity_3'simple_rnn_16/while/Identity_3:output:0"I
simple_rnn_16_while_identity_4'simple_rnn_16/while/Identity_4:output:0"I
simple_rnn_16_while_identity_5'simple_rnn_16/while/Identity_5:output:0"h
1simple_rnn_16_while_simple_rnn_16_strided_slice_13simple_rnn_16_while_simple_rnn_16_strided_slice_1_0"
Fsimple_rnn_16_while_simple_rnn_cell_16_biasadd_readvariableop_resourceHsimple_rnn_16_while_simple_rnn_cell_16_biasadd_readvariableop_resource_0"
Gsimple_rnn_16_while_simple_rnn_cell_16_matmul_1_readvariableop_resourceIsimple_rnn_16_while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0"
Esimple_rnn_16_while_simple_rnn_cell_16_matmul_readvariableop_resourceGsimple_rnn_16_while_simple_rnn_cell_16_matmul_readvariableop_resource_0"ш
qsimple_rnn_16_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_1_tensorlistfromtensorssimple_rnn_16_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_1_tensorlistfromtensor_0"р
msimple_rnn_16_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_tensorlistfromtensorosimple_rnn_16_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : 2~
=simple_rnn_16/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp=simple_rnn_16/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp2|
<simple_rnn_16/while/simple_rnn_cell_16/MatMul/ReadVariableOp<simple_rnn_16/while/simple_rnn_cell_16/MatMul/ReadVariableOp2
>simple_rnn_16/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp>simple_rnn_16/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
м
Њ
while_cond_358228
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_358228___redundant_placeholder04
0while_while_cond_358228___redundant_placeholder14
0while_while_cond_358228___redundant_placeholder24
0while_while_cond_358228___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
і[
Т

while_body_358690
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0M
9while_simple_rnn_cell_17_matmul_readvariableop_resource_0:
I
:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0:	O
;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorK
7while_simple_rnn_cell_17_matmul_readvariableop_resource:
G
8while_simple_rnn_cell_17_biasadd_readvariableop_resource:	M
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource:
Ђ/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpЂ.while/simple_rnn_cell_17/MatMul/ReadVariableOpЂ0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ў
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0

(while/simple_rnn_cell_17/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(while/simple_rnn_cell_17/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?У
"while/simple_rnn_cell_17/ones_likeFill1while/simple_rnn_cell_17/ones_like/Shape:output:01while/simple_rnn_cell_17/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџk
&while/simple_rnn_cell_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @М
$while/simple_rnn_cell_17/dropout/MulMul+while/simple_rnn_cell_17/ones_like:output:0/while/simple_rnn_cell_17/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
&while/simple_rnn_cell_17/dropout/ShapeShape+while/simple_rnn_cell_17/ones_like:output:0*
T0*
_output_shapes
:П
=while/simple_rnn_cell_17/dropout/random_uniform/RandomUniformRandomUniform/while/simple_rnn_cell_17/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0t
/while/simple_rnn_cell_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ђ
-while/simple_rnn_cell_17/dropout/GreaterEqualGreaterEqualFwhile/simple_rnn_cell_17/dropout/random_uniform/RandomUniform:output:08while/simple_rnn_cell_17/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЂ
%while/simple_rnn_cell_17/dropout/CastCast1while/simple_rnn_cell_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЕ
&while/simple_rnn_cell_17/dropout/Mul_1Mul(while/simple_rnn_cell_17/dropout/Mul:z:0)while/simple_rnn_cell_17/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџm
*while/simple_rnn_cell_17/ones_like_1/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:o
*while/simple_rnn_cell_17/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Щ
$while/simple_rnn_cell_17/ones_like_1Fill3while/simple_rnn_cell_17/ones_like_1/Shape:output:03while/simple_rnn_cell_17/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџm
(while/simple_rnn_cell_17/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Т
&while/simple_rnn_cell_17/dropout_1/MulMul-while/simple_rnn_cell_17/ones_like_1:output:01while/simple_rnn_cell_17/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
(while/simple_rnn_cell_17/dropout_1/ShapeShape-while/simple_rnn_cell_17/ones_like_1:output:0*
T0*
_output_shapes
:У
?while/simple_rnn_cell_17/dropout_1/random_uniform/RandomUniformRandomUniform1while/simple_rnn_cell_17/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0v
1while/simple_rnn_cell_17/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ј
/while/simple_rnn_cell_17/dropout_1/GreaterEqualGreaterEqualHwhile/simple_rnn_cell_17/dropout_1/random_uniform/RandomUniform:output:0:while/simple_rnn_cell_17/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџІ
'while/simple_rnn_cell_17/dropout_1/CastCast3while/simple_rnn_cell_17/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЛ
(while/simple_rnn_cell_17/dropout_1/Mul_1Mul*while/simple_rnn_cell_17/dropout_1/Mul:z:0+while/simple_rnn_cell_17/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџД
while/simple_rnn_cell_17/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/simple_rnn_cell_17/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЊ
.while/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_17_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ж
while/simple_rnn_cell_17/MatMulMatMul while/simple_rnn_cell_17/mul:z:06while/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЇ
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Т
 while/simple_rnn_cell_17/BiasAddBiasAdd)while/simple_rnn_cell_17/MatMul:product:07while/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/simple_rnn_cell_17/mul_1Mulwhile_placeholder_3,while/simple_rnn_cell_17/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЎ
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0М
!while/simple_rnn_cell_17/MatMul_1MatMul"while/simple_rnn_cell_17/mul_1:z:08while/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџА
while/simple_rnn_cell_17/addAddV2)while/simple_rnn_cell_17/BiasAdd:output:0+while/simple_rnn_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџz
while/simple_rnn_cell_17/TanhTanh while/simple_rnn_cell_17/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџe
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
while/SelectV2SelectV2while/Tile:output:0!while/simple_rnn_cell_17/Tanh:y:0while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџg
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
while/SelectV2_1SelectV2while/Tile_1:output:0!while/simple_rnn_cell_17/Tanh:y:0while_placeholder_3*
T0*(
_output_shapes
:џџџџџџџџџr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ш
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: u
while/Identity_4Identitywhile/SelectV2:output:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
while/Identity_5Identitywhile/SelectV2_1:output:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџт

while/NoOpNoOp0^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_simple_rnn_cell_17_biasadd_readvariableop_resource:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_17_matmul_readvariableop_resource9while_simple_rnn_cell_17_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"А
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : 2b
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_17/MatMul/ReadVariableOp.while/simple_rnn_cell_17/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
И
­
simple_rnn_16_while_cond_3568298
4simple_rnn_16_while_simple_rnn_16_while_loop_counter>
:simple_rnn_16_while_simple_rnn_16_while_maximum_iterations#
simple_rnn_16_while_placeholder%
!simple_rnn_16_while_placeholder_1%
!simple_rnn_16_while_placeholder_2%
!simple_rnn_16_while_placeholder_3:
6simple_rnn_16_while_less_simple_rnn_16_strided_slice_1P
Lsimple_rnn_16_while_simple_rnn_16_while_cond_356829___redundant_placeholder0P
Lsimple_rnn_16_while_simple_rnn_16_while_cond_356829___redundant_placeholder1P
Lsimple_rnn_16_while_simple_rnn_16_while_cond_356829___redundant_placeholder2P
Lsimple_rnn_16_while_simple_rnn_16_while_cond_356829___redundant_placeholder3P
Lsimple_rnn_16_while_simple_rnn_16_while_cond_356829___redundant_placeholder4 
simple_rnn_16_while_identity

simple_rnn_16/while/LessLesssimple_rnn_16_while_placeholder6simple_rnn_16_while_less_simple_rnn_16_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_16/while/IdentityIdentitysimple_rnn_16/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_16_while_identity%simple_rnn_16/while/Identity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
ќ
Ч
.__inference_simple_rnn_17_layer_call_fn_358178

inputs
mask

unknown:

	unknown_0:	
	unknown_1:

identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsmaskunknown	unknown_0	unknown_1*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_356272p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:џџџџџџџџџ:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:MI
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask
м
Њ
while_cond_357711
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_357711___redundant_placeholder04
0while_while_cond_357711___redundant_placeholder14
0while_while_cond_357711___redundant_placeholder24
0while_while_cond_357711___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
мF
Ш
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_357646
inputs_0D
1simple_rnn_cell_16_matmul_readvariableop_resource:	2A
2simple_rnn_cell_16_biasadd_readvariableop_resource:	G
3simple_rnn_cell_16_matmul_1_readvariableop_resource:

identityЂ)simple_rnn_cell_16/BiasAdd/ReadVariableOpЂ(simple_rnn_cell_16/MatMul/ReadVariableOpЂ*simple_rnn_cell_16/MatMul_1/ReadVariableOpЂwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maskj
"simple_rnn_cell_16/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:g
"simple_rnn_cell_16/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?А
simple_rnn_cell_16/ones_likeFill+simple_rnn_cell_16/ones_like/Shape:output:0+simple_rnn_cell_16/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2b
$simple_rnn_cell_16/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:i
$simple_rnn_cell_16/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?З
simple_rnn_cell_16/ones_like_1Fill-simple_rnn_cell_16/ones_like_1/Shape:output:0-simple_rnn_cell_16/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_16/mulMulstrided_slice_2:output:0%simple_rnn_cell_16/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
(simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_16_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype0Є
simple_rnn_cell_16/MatMulMatMulsimple_rnn_cell_16/mul:z:00simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
)simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0А
simple_rnn_cell_16/BiasAddBiasAdd#simple_rnn_cell_16/MatMul:product:01simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_16/mul_1Mulzeros:output:0'simple_rnn_cell_16/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
*simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Њ
simple_rnn_cell_16/MatMul_1MatMulsimple_rnn_cell_16/mul_1:z:02simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_16/addAddV2#simple_rnn_cell_16/BiasAdd:output:0%simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџn
simple_rnn_cell_16/TanhTanhsimple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_16_matmul_readvariableop_resource2simple_rnn_cell_16_biasadd_readvariableop_resource3simple_rnn_cell_16_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_357572*
condR
while_cond_357571*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ь
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџв
NoOpNoOp*^simple_rnn_cell_16/BiasAdd/ReadVariableOp)^simple_rnn_cell_16/MatMul/ReadVariableOp+^simple_rnn_cell_16/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ2: : : 2V
)simple_rnn_cell_16/BiasAdd/ReadVariableOp)simple_rnn_cell_16/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_16/MatMul/ReadVariableOp(simple_rnn_cell_16/MatMul/ReadVariableOp2X
*simple_rnn_cell_16/MatMul_1/ReadVariableOp*simple_rnn_cell_16/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
"
_user_specified_name
inputs/0
їd
а
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_358132

inputs
mask
D
1simple_rnn_cell_16_matmul_readvariableop_resource:	2A
2simple_rnn_cell_16_biasadd_readvariableop_resource:	G
3simple_rnn_cell_16_matmul_1_readvariableop_resource:

identityЂ)simple_rnn_cell_16/BiasAdd/ReadVariableOpЂ(simple_rnn_cell_16/MatMul/ReadVariableOpЂ*simple_rnn_cell_16/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџm

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*+
_output_shapes
:џџџџџџџџџe
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ~
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*+
_output_shapes
:џџџџџџџџџf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maskj
"simple_rnn_cell_16/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:g
"simple_rnn_cell_16/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?А
simple_rnn_cell_16/ones_likeFill+simple_rnn_cell_16/ones_like/Shape:output:0+simple_rnn_cell_16/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2e
 simple_rnn_cell_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Љ
simple_rnn_cell_16/dropout/MulMul%simple_rnn_cell_16/ones_like:output:0)simple_rnn_cell_16/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2u
 simple_rnn_cell_16/dropout/ShapeShape%simple_rnn_cell_16/ones_like:output:0*
T0*
_output_shapes
:В
7simple_rnn_cell_16/dropout/random_uniform/RandomUniformRandomUniform)simple_rnn_cell_16/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype0n
)simple_rnn_cell_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?п
'simple_rnn_cell_16/dropout/GreaterEqualGreaterEqual@simple_rnn_cell_16/dropout/random_uniform/RandomUniform:output:02simple_rnn_cell_16/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_16/dropout/CastCast+simple_rnn_cell_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2Ђ
 simple_rnn_cell_16/dropout/Mul_1Mul"simple_rnn_cell_16/dropout/Mul:z:0#simple_rnn_cell_16/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2b
$simple_rnn_cell_16/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:i
$simple_rnn_cell_16/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?З
simple_rnn_cell_16/ones_like_1Fill-simple_rnn_cell_16/ones_like_1/Shape:output:0-simple_rnn_cell_16/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџg
"simple_rnn_cell_16/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @А
 simple_rnn_cell_16/dropout_1/MulMul'simple_rnn_cell_16/ones_like_1:output:0+simple_rnn_cell_16/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџy
"simple_rnn_cell_16/dropout_1/ShapeShape'simple_rnn_cell_16/ones_like_1:output:0*
T0*
_output_shapes
:З
9simple_rnn_cell_16/dropout_1/random_uniform/RandomUniformRandomUniform+simple_rnn_cell_16/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0p
+simple_rnn_cell_16/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ц
)simple_rnn_cell_16/dropout_1/GreaterEqualGreaterEqualBsimple_rnn_cell_16/dropout_1/random_uniform/RandomUniform:output:04simple_rnn_cell_16/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
!simple_rnn_cell_16/dropout_1/CastCast-simple_rnn_cell_16/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЉ
"simple_rnn_cell_16/dropout_1/Mul_1Mul$simple_rnn_cell_16/dropout_1/Mul:z:0%simple_rnn_cell_16/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_16/mulMulstrided_slice_2:output:0$simple_rnn_cell_16/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
(simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_16_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype0Є
simple_rnn_cell_16/MatMulMatMulsimple_rnn_cell_16/mul:z:00simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
)simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0А
simple_rnn_cell_16/BiasAddBiasAdd#simple_rnn_cell_16/MatMul:product:01simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_16/mul_1Mulzeros:output:0&simple_rnn_cell_16/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ 
*simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Њ
simple_rnn_cell_16/MatMul_1MatMulsimple_rnn_cell_16/mul_1:z:02simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_16/addAddV2#simple_rnn_cell_16/BiasAdd:output:0%simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџn
simple_rnn_cell_16/TanhTanhsimple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : h
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџИ
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ц
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШg

zeros_like	ZerosLikesimple_rnn_cell_16/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџc
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ж
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:01simple_rnn_cell_16_matmul_readvariableop_resource2simple_rnn_cell_16_biasadd_readvariableop_resource3simple_rnn_cell_16_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*P
_output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_358027*
condR
while_cond_358026*O
output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   У
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџc
IdentityIdentitytranspose_2:y:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџв
NoOpNoOp*^simple_rnn_cell_16/BiasAdd/ReadVariableOp)^simple_rnn_cell_16/MatMul/ReadVariableOp+^simple_rnn_cell_16/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџ2:џџџџџџџџџ: : : 2V
)simple_rnn_cell_16/BiasAdd/ReadVariableOp)simple_rnn_cell_16/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_16/MatMul/ReadVariableOp(simple_rnn_cell_16/MatMul/ReadVariableOp2X
*simple_rnn_cell_16/MatMul_1/ReadVariableOp*simple_rnn_cell_16/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs:MI
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask
Х

)__inference_dense_24_layer_call_fn_358805

inputs
unknown:	@
	unknown_0:@
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_356005o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Р$
я
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_358930

inputs
states_01
matmul_readvariableop_resource:	2.
biasadd_readvariableop_resource:	4
 matmul_1_readvariableop_resource:

identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2I
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @w
dropout_1/MulMulones_like_1:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџS
dropout_1/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџt
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџp
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџW
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2*
dtype0k
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ^
mul_1Mulstates_0dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
MatMul_1MatMul	mul_1:z:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџH
TanhTanhadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџX
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџZ

Identity_1IdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџ2:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
ЯG
Щ
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_358304
inputs_0E
1simple_rnn_cell_17_matmul_readvariableop_resource:
A
2simple_rnn_cell_17_biasadd_readvariableop_resource:	G
3simple_rnn_cell_17_matmul_1_readvariableop_resource:

identityЂ)simple_rnn_cell_17/BiasAdd/ReadVariableOpЂ(simple_rnn_cell_17/MatMul/ReadVariableOpЂ*simple_rnn_cell_17/MatMul_1/ReadVariableOpЂwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskj
"simple_rnn_cell_17/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:g
"simple_rnn_cell_17/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Б
simple_rnn_cell_17/ones_likeFill+simple_rnn_cell_17/ones_like/Shape:output:0+simple_rnn_cell_17/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
$simple_rnn_cell_17/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:i
$simple_rnn_cell_17/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?З
simple_rnn_cell_17/ones_like_1Fill-simple_rnn_cell_17/ones_like_1/Shape:output:0-simple_rnn_cell_17/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/mulMulstrided_slice_2:output:0%simple_rnn_cell_17/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
(simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_17_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Є
simple_rnn_cell_17/MatMulMatMulsimple_rnn_cell_17/mul:z:00simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
)simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0А
simple_rnn_cell_17/BiasAddBiasAdd#simple_rnn_cell_17/MatMul:product:01simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/mul_1Mulzeros:output:0'simple_rnn_cell_17/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
*simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_17_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Њ
simple_rnn_cell_17/MatMul_1MatMulsimple_rnn_cell_17/mul_1:z:02simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/addAddV2#simple_rnn_cell_17/BiasAdd:output:0%simple_rnn_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџn
simple_rnn_cell_17/TanhTanhsimple_rnn_cell_17/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_17_matmul_readvariableop_resource2simple_rnn_cell_17_biasadd_readvariableop_resource3simple_rnn_cell_17_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_358229*
condR
while_cond_358228*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   з
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџв
NoOpNoOp*^simple_rnn_cell_17/BiasAdd/ReadVariableOp)^simple_rnn_cell_17/MatMul/ReadVariableOp+^simple_rnn_cell_17/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџ: : : 2V
)simple_rnn_cell_17/BiasAdd/ReadVariableOp)simple_rnn_cell_17/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_17/MatMul/ReadVariableOp(simple_rnn_cell_17/MatMul/ReadVariableOp2X
*simple_rnn_cell_17/MatMul_1/ReadVariableOp*simple_rnn_cell_17/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Њ6
й
while_body_357572
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_16_matmul_readvariableop_resource_0:	2I
:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0:	O
;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_16_matmul_readvariableop_resource:	2G
8while_simple_rnn_cell_16_biasadd_readvariableop_resource:	M
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource:
Ђ/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpЂ.while/simple_rnn_cell_16/MatMul/ReadVariableOpЂ0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ2*
element_dtype0
(while/simple_rnn_cell_16/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(while/simple_rnn_cell_16/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Т
"while/simple_rnn_cell_16/ones_likeFill1while/simple_rnn_cell_16/ones_like/Shape:output:01while/simple_rnn_cell_16/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2m
*while/simple_rnn_cell_16/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:o
*while/simple_rnn_cell_16/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Щ
$while/simple_rnn_cell_16/ones_like_1Fill3while/simple_rnn_cell_16/ones_like_1/Shape:output:03while/simple_rnn_cell_16/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџД
while/simple_rnn_cell_16/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/simple_rnn_cell_16/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2Љ
.while/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	2*
dtype0Ж
while/simple_rnn_cell_16/MatMulMatMul while/simple_rnn_cell_16/mul:z:06while/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЇ
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Т
 while/simple_rnn_cell_16/BiasAddBiasAdd)while/simple_rnn_cell_16/MatMul:product:07while/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/simple_rnn_cell_16/mul_1Mulwhile_placeholder_2-while/simple_rnn_cell_16/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЎ
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0М
!while/simple_rnn_cell_16/MatMul_1MatMul"while/simple_rnn_cell_16/mul_1:z:08while/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџА
while/simple_rnn_cell_16/addAddV2)while/simple_rnn_cell_16/BiasAdd:output:0+while/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџz
while/simple_rnn_cell_16/TanhTanh while/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЪ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_16/Tanh:y:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity!while/simple_rnn_cell_16/Tanh:y:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџт

while/NoOpNoOp0^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_16_biasadd_readvariableop_resource:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_16_matmul_readvariableop_resource9while_simple_rnn_cell_16_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџ: : : : : 2b
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_16/MatMul/ReadVariableOp.while/simple_rnn_cell_16/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
И
­
simple_rnn_17_while_cond_3569748
4simple_rnn_17_while_simple_rnn_17_while_loop_counter>
:simple_rnn_17_while_simple_rnn_17_while_maximum_iterations#
simple_rnn_17_while_placeholder%
!simple_rnn_17_while_placeholder_1%
!simple_rnn_17_while_placeholder_2%
!simple_rnn_17_while_placeholder_3:
6simple_rnn_17_while_less_simple_rnn_17_strided_slice_1P
Lsimple_rnn_17_while_simple_rnn_17_while_cond_356974___redundant_placeholder0P
Lsimple_rnn_17_while_simple_rnn_17_while_cond_356974___redundant_placeholder1P
Lsimple_rnn_17_while_simple_rnn_17_while_cond_356974___redundant_placeholder2P
Lsimple_rnn_17_while_simple_rnn_17_while_cond_356974___redundant_placeholder3P
Lsimple_rnn_17_while_simple_rnn_17_while_cond_356974___redundant_placeholder4 
simple_rnn_17_while_identity

simple_rnn_17/while/LessLesssimple_rnn_17_while_placeholder6simple_rnn_17_while_less_simple_rnn_17_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_17/while/IdentityIdentitysimple_rnn_17/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_17_while_identity%simple_rnn_17/while/Identity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
И
О
.__inference_simple_rnn_16_layer_call_fn_357487
inputs_0
unknown:	2
	unknown_0:	
	unknown_1:

identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_355137}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
"
_user_specified_name
inputs/0


і
D__inference_dense_24_layer_call_and_return_conditional_losses_358816

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	
љ
while_cond_356371
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_356371___redundant_placeholder04
0while_while_cond_356371___redundant_placeholder14
0while_while_cond_356371___redundant_placeholder24
0while_while_cond_356371___redundant_placeholder34
0while_while_cond_356371___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
ФR
б
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_358613

inputs
mask
E
1simple_rnn_cell_17_matmul_readvariableop_resource:
A
2simple_rnn_cell_17_biasadd_readvariableop_resource:	G
3simple_rnn_cell_17_matmul_1_readvariableop_resource:

identityЂ)simple_rnn_cell_17/BiasAdd/ReadVariableOpЂ(simple_rnn_cell_17/MatMul/ReadVariableOpЂ*simple_rnn_cell_17/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџm

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*+
_output_shapes
:џџџџџџџџџe
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ~
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*+
_output_shapes
:џџџџџџџџџf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskj
"simple_rnn_cell_17/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:g
"simple_rnn_cell_17/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Б
simple_rnn_cell_17/ones_likeFill+simple_rnn_cell_17/ones_like/Shape:output:0+simple_rnn_cell_17/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
$simple_rnn_cell_17/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:i
$simple_rnn_cell_17/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?З
simple_rnn_cell_17/ones_like_1Fill-simple_rnn_cell_17/ones_like_1/Shape:output:0-simple_rnn_cell_17/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/mulMulstrided_slice_2:output:0%simple_rnn_cell_17/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
(simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_17_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Є
simple_rnn_cell_17/MatMulMatMulsimple_rnn_cell_17/mul:z:00simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
)simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0А
simple_rnn_cell_17/BiasAddBiasAdd#simple_rnn_cell_17/MatMul:product:01simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/mul_1Mulzeros:output:0'simple_rnn_cell_17/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
*simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_17_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Њ
simple_rnn_cell_17/MatMul_1MatMulsimple_rnn_cell_17/mul_1:z:02simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/addAddV2#simple_rnn_cell_17/BiasAdd:output:0%simple_rnn_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџn
simple_rnn_cell_17/TanhTanhsimple_rnn_cell_17/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : h
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџИ
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ц
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШg

zeros_like	ZerosLikesimple_rnn_cell_17/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџc
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ж
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:01simple_rnn_cell_17_matmul_readvariableop_resource2simple_rnn_cell_17_biasadd_readvariableop_resource3simple_rnn_cell_17_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*P
_output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_358523*
condR
while_cond_358522*O
output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   з
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџв
NoOpNoOp*^simple_rnn_cell_17/BiasAdd/ReadVariableOp)^simple_rnn_cell_17/MatMul/ReadVariableOp+^simple_rnn_cell_17/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:џџџџџџџџџ:џџџџџџџџџ: : : 2V
)simple_rnn_cell_17/BiasAdd/ReadVariableOp)simple_rnn_cell_17/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_17/MatMul/ReadVariableOp(simple_rnn_cell_17/MatMul/ReadVariableOp2X
*simple_rnn_cell_17/MatMul_1/ReadVariableOp*simple_rnn_cell_17/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:MI
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask
Х

п
3__inference_simple_rnn_cell_16_layer_call_fn_358850

inputs
states_0
unknown:	2
	unknown_0:	
	unknown_1:

identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_355061p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџ2:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
[
Щ
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_358462
inputs_0E
1simple_rnn_cell_17_matmul_readvariableop_resource:
A
2simple_rnn_cell_17_biasadd_readvariableop_resource:	G
3simple_rnn_cell_17_matmul_1_readvariableop_resource:

identityЂ)simple_rnn_cell_17/BiasAdd/ReadVariableOpЂ(simple_rnn_cell_17/MatMul/ReadVariableOpЂ*simple_rnn_cell_17/MatMul_1/ReadVariableOpЂwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskj
"simple_rnn_cell_17/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:g
"simple_rnn_cell_17/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Б
simple_rnn_cell_17/ones_likeFill+simple_rnn_cell_17/ones_like/Shape:output:0+simple_rnn_cell_17/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџe
 simple_rnn_cell_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Њ
simple_rnn_cell_17/dropout/MulMul%simple_rnn_cell_17/ones_like:output:0)simple_rnn_cell_17/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџu
 simple_rnn_cell_17/dropout/ShapeShape%simple_rnn_cell_17/ones_like:output:0*
T0*
_output_shapes
:Г
7simple_rnn_cell_17/dropout/random_uniform/RandomUniformRandomUniform)simple_rnn_cell_17/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0n
)simple_rnn_cell_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?р
'simple_rnn_cell_17/dropout/GreaterEqualGreaterEqual@simple_rnn_cell_17/dropout/random_uniform/RandomUniform:output:02simple_rnn_cell_17/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/dropout/CastCast+simple_rnn_cell_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЃ
 simple_rnn_cell_17/dropout/Mul_1Mul"simple_rnn_cell_17/dropout/Mul:z:0#simple_rnn_cell_17/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџb
$simple_rnn_cell_17/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:i
$simple_rnn_cell_17/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?З
simple_rnn_cell_17/ones_like_1Fill-simple_rnn_cell_17/ones_like_1/Shape:output:0-simple_rnn_cell_17/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџg
"simple_rnn_cell_17/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @А
 simple_rnn_cell_17/dropout_1/MulMul'simple_rnn_cell_17/ones_like_1:output:0+simple_rnn_cell_17/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџy
"simple_rnn_cell_17/dropout_1/ShapeShape'simple_rnn_cell_17/ones_like_1:output:0*
T0*
_output_shapes
:З
9simple_rnn_cell_17/dropout_1/random_uniform/RandomUniformRandomUniform+simple_rnn_cell_17/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0p
+simple_rnn_cell_17/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ц
)simple_rnn_cell_17/dropout_1/GreaterEqualGreaterEqualBsimple_rnn_cell_17/dropout_1/random_uniform/RandomUniform:output:04simple_rnn_cell_17/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
!simple_rnn_cell_17/dropout_1/CastCast-simple_rnn_cell_17/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЉ
"simple_rnn_cell_17/dropout_1/Mul_1Mul$simple_rnn_cell_17/dropout_1/Mul:z:0%simple_rnn_cell_17/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/mulMulstrided_slice_2:output:0$simple_rnn_cell_17/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
(simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_17_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Є
simple_rnn_cell_17/MatMulMatMulsimple_rnn_cell_17/mul:z:00simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
)simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0А
simple_rnn_cell_17/BiasAddBiasAdd#simple_rnn_cell_17/MatMul:product:01simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/mul_1Mulzeros:output:0&simple_rnn_cell_17/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ 
*simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_17_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Њ
simple_rnn_cell_17/MatMul_1MatMulsimple_rnn_cell_17/mul_1:z:02simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/addAddV2#simple_rnn_cell_17/BiasAdd:output:0%simple_rnn_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџn
simple_rnn_cell_17/TanhTanhsimple_rnn_cell_17/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_17_matmul_readvariableop_resource2simple_rnn_cell_17_biasadd_readvariableop_resource3simple_rnn_cell_17_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_358371*
condR
while_cond_358370*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   з
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџв
NoOpNoOp*^simple_rnn_cell_17/BiasAdd/ReadVariableOp)^simple_rnn_cell_17/MatMul/ReadVariableOp+^simple_rnn_cell_17/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџ: : : 2V
)simple_rnn_cell_17/BiasAdd/ReadVariableOp)simple_rnn_cell_17/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_17/MatMul/ReadVariableOp(simple_rnn_cell_17/MatMul/ReadVariableOp2X
*simple_rnn_cell_17/MatMul_1/ReadVariableOp*simple_rnn_cell_17/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
н
с
-sequential_12_simple_rnn_16_while_cond_354755T
Psequential_12_simple_rnn_16_while_sequential_12_simple_rnn_16_while_loop_counterZ
Vsequential_12_simple_rnn_16_while_sequential_12_simple_rnn_16_while_maximum_iterations1
-sequential_12_simple_rnn_16_while_placeholder3
/sequential_12_simple_rnn_16_while_placeholder_13
/sequential_12_simple_rnn_16_while_placeholder_23
/sequential_12_simple_rnn_16_while_placeholder_3V
Rsequential_12_simple_rnn_16_while_less_sequential_12_simple_rnn_16_strided_slice_1l
hsequential_12_simple_rnn_16_while_sequential_12_simple_rnn_16_while_cond_354755___redundant_placeholder0l
hsequential_12_simple_rnn_16_while_sequential_12_simple_rnn_16_while_cond_354755___redundant_placeholder1l
hsequential_12_simple_rnn_16_while_sequential_12_simple_rnn_16_while_cond_354755___redundant_placeholder2l
hsequential_12_simple_rnn_16_while_sequential_12_simple_rnn_16_while_cond_354755___redundant_placeholder3l
hsequential_12_simple_rnn_16_while_sequential_12_simple_rnn_16_while_cond_354755___redundant_placeholder4.
*sequential_12_simple_rnn_16_while_identity
в
&sequential_12/simple_rnn_16/while/LessLess-sequential_12_simple_rnn_16_while_placeholderRsequential_12_simple_rnn_16_while_less_sequential_12_simple_rnn_16_strided_slice_1*
T0*
_output_shapes
: 
*sequential_12/simple_rnn_16/while/IdentityIdentity*sequential_12/simple_rnn_16/while/Less:z:0*
T0
*
_output_shapes
: "a
*sequential_12_simple_rnn_16_while_identity3sequential_12/simple_rnn_16/while/Identity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
ыл
Џ
I__inference_sequential_12_layer_call_and_return_conditional_losses_357079

inputs7
$embedding_12_embedding_lookup_356767:	'2R
?simple_rnn_16_simple_rnn_cell_16_matmul_readvariableop_resource:	2O
@simple_rnn_16_simple_rnn_cell_16_biasadd_readvariableop_resource:	U
Asimple_rnn_16_simple_rnn_cell_16_matmul_1_readvariableop_resource:
S
?simple_rnn_17_simple_rnn_cell_17_matmul_readvariableop_resource:
O
@simple_rnn_17_simple_rnn_cell_17_biasadd_readvariableop_resource:	U
Asimple_rnn_17_simple_rnn_cell_17_matmul_1_readvariableop_resource:
:
'dense_24_matmul_readvariableop_resource:	@6
(dense_24_biasadd_readvariableop_resource:@9
'dense_25_matmul_readvariableop_resource:@6
(dense_25_biasadd_readvariableop_resource:
identityЂdense_24/BiasAdd/ReadVariableOpЂdense_24/MatMul/ReadVariableOpЂdense_25/BiasAdd/ReadVariableOpЂdense_25/MatMul/ReadVariableOpЂembedding_12/embedding_lookupЂ7simple_rnn_16/simple_rnn_cell_16/BiasAdd/ReadVariableOpЂ6simple_rnn_16/simple_rnn_cell_16/MatMul/ReadVariableOpЂ8simple_rnn_16/simple_rnn_cell_16/MatMul_1/ReadVariableOpЂsimple_rnn_16/whileЂ7simple_rnn_17/simple_rnn_cell_17/BiasAdd/ReadVariableOpЂ6simple_rnn_17/simple_rnn_cell_17/MatMul/ReadVariableOpЂ8simple_rnn_17/simple_rnn_cell_17/MatMul_1/ReadVariableOpЂsimple_rnn_17/whileb
embedding_12/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџя
embedding_12/embedding_lookupResourceGather$embedding_12_embedding_lookup_356767embedding_12/Cast:y:0*
Tindices0*7
_class-
+)loc:@embedding_12/embedding_lookup/356767*+
_output_shapes
:џџџџџџџџџ2*
dtype0Щ
&embedding_12/embedding_lookup/IdentityIdentity&embedding_12/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_12/embedding_lookup/356767*+
_output_shapes
:џџџџџџџџџ2
(embedding_12/embedding_lookup/Identity_1Identity/embedding_12/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2\
embedding_12/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
embedding_12/NotEqualNotEqualinputs embedding_12/NotEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџt
simple_rnn_16/ShapeShape1embedding_12/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:k
!simple_rnn_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_16/strided_sliceStridedSlicesimple_rnn_16/Shape:output:0*simple_rnn_16/strided_slice/stack:output:0,simple_rnn_16/strided_slice/stack_1:output:0,simple_rnn_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
simple_rnn_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
simple_rnn_16/zeros/packedPack$simple_rnn_16/strided_slice:output:0%simple_rnn_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
simple_rnn_16/zerosFill#simple_rnn_16/zeros/packed:output:0"simple_rnn_16/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџq
simple_rnn_16/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Д
simple_rnn_16/transpose	Transpose1embedding_12/embedding_lookup/Identity_1:output:0%simple_rnn_16/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2`
simple_rnn_16/Shape_1Shapesimple_rnn_16/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
simple_rnn_16/strided_slice_1StridedSlicesimple_rnn_16/Shape_1:output:0,simple_rnn_16/strided_slice_1/stack:output:0.simple_rnn_16/strided_slice_1/stack_1:output:0.simple_rnn_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
simple_rnn_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
simple_rnn_16/ExpandDims
ExpandDimsembedding_12/NotEqual:z:0%simple_rnn_16/ExpandDims/dim:output:0*
T0
*+
_output_shapes
:џџџџџџџџџs
simple_rnn_16/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ј
simple_rnn_16/transpose_1	Transpose!simple_rnn_16/ExpandDims:output:0'simple_rnn_16/transpose_1/perm:output:0*
T0
*+
_output_shapes
:џџџџџџџџџt
)simple_rnn_16/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџо
simple_rnn_16/TensorArrayV2TensorListReserve2simple_rnn_16/TensorArrayV2/element_shape:output:0&simple_rnn_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Csimple_rnn_16/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   
5simple_rnn_16/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_16/transpose:y:0Lsimple_rnn_16/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвm
#simple_rnn_16/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_16/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_16/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Џ
simple_rnn_16/strided_slice_2StridedSlicesimple_rnn_16/transpose:y:0,simple_rnn_16/strided_slice_2/stack:output:0.simple_rnn_16/strided_slice_2/stack_1:output:0.simple_rnn_16/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask
0simple_rnn_16/simple_rnn_cell_16/ones_like/ShapeShape&simple_rnn_16/strided_slice_2:output:0*
T0*
_output_shapes
:u
0simple_rnn_16/simple_rnn_cell_16/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?к
*simple_rnn_16/simple_rnn_cell_16/ones_likeFill9simple_rnn_16/simple_rnn_cell_16/ones_like/Shape:output:09simple_rnn_16/simple_rnn_cell_16/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2~
2simple_rnn_16/simple_rnn_cell_16/ones_like_1/ShapeShapesimple_rnn_16/zeros:output:0*
T0*
_output_shapes
:w
2simple_rnn_16/simple_rnn_cell_16/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?с
,simple_rnn_16/simple_rnn_cell_16/ones_like_1Fill;simple_rnn_16/simple_rnn_cell_16/ones_like_1/Shape:output:0;simple_rnn_16/simple_rnn_cell_16/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџК
$simple_rnn_16/simple_rnn_cell_16/mulMul&simple_rnn_16/strided_slice_2:output:03simple_rnn_16/simple_rnn_cell_16/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2З
6simple_rnn_16/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp?simple_rnn_16_simple_rnn_cell_16_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype0Ю
'simple_rnn_16/simple_rnn_cell_16/MatMulMatMul(simple_rnn_16/simple_rnn_cell_16/mul:z:0>simple_rnn_16/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЕ
7simple_rnn_16/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_16_simple_rnn_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0к
(simple_rnn_16/simple_rnn_cell_16/BiasAddBiasAdd1simple_rnn_16/simple_rnn_cell_16/MatMul:product:0?simple_rnn_16/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЕ
&simple_rnn_16/simple_rnn_cell_16/mul_1Mulsimple_rnn_16/zeros:output:05simple_rnn_16/simple_rnn_cell_16/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџМ
8simple_rnn_16/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_16_simple_rnn_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0д
)simple_rnn_16/simple_rnn_cell_16/MatMul_1MatMul*simple_rnn_16/simple_rnn_cell_16/mul_1:z:0@simple_rnn_16/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ
$simple_rnn_16/simple_rnn_cell_16/addAddV21simple_rnn_16/simple_rnn_cell_16/BiasAdd:output:03simple_rnn_16/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
%simple_rnn_16/simple_rnn_cell_16/TanhTanh(simple_rnn_16/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ|
+simple_rnn_16/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   т
simple_rnn_16/TensorArrayV2_1TensorListReserve4simple_rnn_16/TensorArrayV2_1/element_shape:output:0&simple_rnn_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвT
simple_rnn_16/timeConst*
_output_shapes
: *
dtype0*
value	B : v
+simple_rnn_16/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџт
simple_rnn_16/TensorArrayV2_2TensorListReserve4simple_rnn_16/TensorArrayV2_2/element_shape:output:0&simple_rnn_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
Esimple_rnn_16/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
7simple_rnn_16/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorsimple_rnn_16/transpose_1:y:0Nsimple_rnn_16/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
simple_rnn_16/zeros_like	ZerosLike)simple_rnn_16/simple_rnn_cell_16/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџq
&simple_rnn_16/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџb
 simple_rnn_16/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ј
simple_rnn_16/whileWhile)simple_rnn_16/while/loop_counter:output:0/simple_rnn_16/while/maximum_iterations:output:0simple_rnn_16/time:output:0&simple_rnn_16/TensorArrayV2_1:handle:0simple_rnn_16/zeros_like:y:0simple_rnn_16/zeros:output:0&simple_rnn_16/strided_slice_1:output:0Esimple_rnn_16/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gsimple_rnn_16/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0?simple_rnn_16_simple_rnn_cell_16_matmul_readvariableop_resource@simple_rnn_16_simple_rnn_cell_16_biasadd_readvariableop_resourceAsimple_rnn_16_simple_rnn_cell_16_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*P
_output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *+
body#R!
simple_rnn_16_while_body_356830*+
cond#R!
simple_rnn_16_while_cond_356829*O
output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *
parallel_iterations 
>simple_rnn_16/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   э
0simple_rnn_16/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_16/while:output:3Gsimple_rnn_16/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0v
#simple_rnn_16/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџo
%simple_rnn_16/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_16/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ю
simple_rnn_16/strided_slice_3StridedSlice9simple_rnn_16/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_16/strided_slice_3/stack:output:0.simple_rnn_16/strided_slice_3/stack_1:output:0.simple_rnn_16/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_masks
simple_rnn_16/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          С
simple_rnn_16/transpose_2	Transpose9simple_rnn_16/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_16/transpose_2/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ`
simple_rnn_17/ShapeShapesimple_rnn_16/transpose_2:y:0*
T0*
_output_shapes
:k
!simple_rnn_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_17/strided_sliceStridedSlicesimple_rnn_17/Shape:output:0*simple_rnn_17/strided_slice/stack:output:0,simple_rnn_17/strided_slice/stack_1:output:0,simple_rnn_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
simple_rnn_17/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
simple_rnn_17/zeros/packedPack$simple_rnn_17/strided_slice:output:0%simple_rnn_17/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_17/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
simple_rnn_17/zerosFill#simple_rnn_17/zeros/packed:output:0"simple_rnn_17/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџq
simple_rnn_17/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ё
simple_rnn_17/transpose	Transposesimple_rnn_16/transpose_2:y:0%simple_rnn_17/transpose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ`
simple_rnn_17/Shape_1Shapesimple_rnn_17/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
simple_rnn_17/strided_slice_1StridedSlicesimple_rnn_17/Shape_1:output:0,simple_rnn_17/strided_slice_1/stack:output:0.simple_rnn_17/strided_slice_1/stack_1:output:0.simple_rnn_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
simple_rnn_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
simple_rnn_17/ExpandDims
ExpandDimsembedding_12/NotEqual:z:0%simple_rnn_17/ExpandDims/dim:output:0*
T0
*+
_output_shapes
:џџџџџџџџџs
simple_rnn_17/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ј
simple_rnn_17/transpose_1	Transpose!simple_rnn_17/ExpandDims:output:0'simple_rnn_17/transpose_1/perm:output:0*
T0
*+
_output_shapes
:џџџџџџџџџt
)simple_rnn_17/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџо
simple_rnn_17/TensorArrayV2TensorListReserve2simple_rnn_17/TensorArrayV2/element_shape:output:0&simple_rnn_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Csimple_rnn_17/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
5simple_rnn_17/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_17/transpose:y:0Lsimple_rnn_17/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвm
#simple_rnn_17/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_17/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_17/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
simple_rnn_17/strided_slice_2StridedSlicesimple_rnn_17/transpose:y:0,simple_rnn_17/strided_slice_2/stack:output:0.simple_rnn_17/strided_slice_2/stack_1:output:0.simple_rnn_17/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
0simple_rnn_17/simple_rnn_cell_17/ones_like/ShapeShape&simple_rnn_17/strided_slice_2:output:0*
T0*
_output_shapes
:u
0simple_rnn_17/simple_rnn_cell_17/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?л
*simple_rnn_17/simple_rnn_cell_17/ones_likeFill9simple_rnn_17/simple_rnn_cell_17/ones_like/Shape:output:09simple_rnn_17/simple_rnn_cell_17/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ~
2simple_rnn_17/simple_rnn_cell_17/ones_like_1/ShapeShapesimple_rnn_17/zeros:output:0*
T0*
_output_shapes
:w
2simple_rnn_17/simple_rnn_cell_17/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?с
,simple_rnn_17/simple_rnn_cell_17/ones_like_1Fill;simple_rnn_17/simple_rnn_cell_17/ones_like_1/Shape:output:0;simple_rnn_17/simple_rnn_cell_17/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЛ
$simple_rnn_17/simple_rnn_cell_17/mulMul&simple_rnn_17/strided_slice_2:output:03simple_rnn_17/simple_rnn_cell_17/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџИ
6simple_rnn_17/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp?simple_rnn_17_simple_rnn_cell_17_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ю
'simple_rnn_17/simple_rnn_cell_17/MatMulMatMul(simple_rnn_17/simple_rnn_cell_17/mul:z:0>simple_rnn_17/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЕ
7simple_rnn_17/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_17_simple_rnn_cell_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0к
(simple_rnn_17/simple_rnn_cell_17/BiasAddBiasAdd1simple_rnn_17/simple_rnn_cell_17/MatMul:product:0?simple_rnn_17/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЕ
&simple_rnn_17/simple_rnn_cell_17/mul_1Mulsimple_rnn_17/zeros:output:05simple_rnn_17/simple_rnn_cell_17/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџМ
8simple_rnn_17/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_17_simple_rnn_cell_17_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0д
)simple_rnn_17/simple_rnn_cell_17/MatMul_1MatMul*simple_rnn_17/simple_rnn_cell_17/mul_1:z:0@simple_rnn_17/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ
$simple_rnn_17/simple_rnn_cell_17/addAddV21simple_rnn_17/simple_rnn_cell_17/BiasAdd:output:03simple_rnn_17/simple_rnn_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
%simple_rnn_17/simple_rnn_cell_17/TanhTanh(simple_rnn_17/simple_rnn_cell_17/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ|
+simple_rnn_17/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   l
*simple_rnn_17/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :я
simple_rnn_17/TensorArrayV2_1TensorListReserve4simple_rnn_17/TensorArrayV2_1/element_shape:output:03simple_rnn_17/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвT
simple_rnn_17/timeConst*
_output_shapes
: *
dtype0*
value	B : v
+simple_rnn_17/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџт
simple_rnn_17/TensorArrayV2_2TensorListReserve4simple_rnn_17/TensorArrayV2_2/element_shape:output:0&simple_rnn_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
Esimple_rnn_17/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
7simple_rnn_17/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorsimple_rnn_17/transpose_1:y:0Nsimple_rnn_17/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
simple_rnn_17/zeros_like	ZerosLike)simple_rnn_17/simple_rnn_cell_17/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџq
&simple_rnn_17/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџb
 simple_rnn_17/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ј
simple_rnn_17/whileWhile)simple_rnn_17/while/loop_counter:output:0/simple_rnn_17/while/maximum_iterations:output:0simple_rnn_17/time:output:0&simple_rnn_17/TensorArrayV2_1:handle:0simple_rnn_17/zeros_like:y:0simple_rnn_17/zeros:output:0&simple_rnn_17/strided_slice_1:output:0Esimple_rnn_17/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gsimple_rnn_17/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0?simple_rnn_17_simple_rnn_cell_17_matmul_readvariableop_resource@simple_rnn_17_simple_rnn_cell_17_biasadd_readvariableop_resourceAsimple_rnn_17_simple_rnn_cell_17_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*P
_output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *+
body#R!
simple_rnn_17_while_body_356975*+
cond#R!
simple_rnn_17_while_cond_356974*O
output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *
parallel_iterations 
>simple_rnn_17/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
0simple_rnn_17/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_17/while:output:3Gsimple_rnn_17/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsv
#simple_rnn_17/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџo
%simple_rnn_17/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_17/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ю
simple_rnn_17/strided_slice_3StridedSlice9simple_rnn_17/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_17/strided_slice_3/stack:output:0.simple_rnn_17/strided_slice_3/stack_1:output:0.simple_rnn_17/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_masks
simple_rnn_17/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          С
simple_rnn_17/transpose_2	Transpose9simple_rnn_17/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_17/transpose_2/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_24/MatMulMatMul&simple_rnn_17/strided_slice_3:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@b
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_25/MatMulMatMuldense_24/Relu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџh
dense_25/SigmoidSigmoiddense_25/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
IdentityIdentitydense_25/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџє
NoOpNoOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp^embedding_12/embedding_lookup8^simple_rnn_16/simple_rnn_cell_16/BiasAdd/ReadVariableOp7^simple_rnn_16/simple_rnn_cell_16/MatMul/ReadVariableOp9^simple_rnn_16/simple_rnn_cell_16/MatMul_1/ReadVariableOp^simple_rnn_16/while8^simple_rnn_17/simple_rnn_cell_17/BiasAdd/ReadVariableOp7^simple_rnn_17/simple_rnn_cell_17/MatMul/ReadVariableOp9^simple_rnn_17/simple_rnn_cell_17/MatMul_1/ReadVariableOp^simple_rnn_17/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2>
embedding_12/embedding_lookupembedding_12/embedding_lookup2r
7simple_rnn_16/simple_rnn_cell_16/BiasAdd/ReadVariableOp7simple_rnn_16/simple_rnn_cell_16/BiasAdd/ReadVariableOp2p
6simple_rnn_16/simple_rnn_cell_16/MatMul/ReadVariableOp6simple_rnn_16/simple_rnn_cell_16/MatMul/ReadVariableOp2t
8simple_rnn_16/simple_rnn_cell_16/MatMul_1/ReadVariableOp8simple_rnn_16/simple_rnn_cell_16/MatMul_1/ReadVariableOp2*
simple_rnn_16/whilesimple_rnn_16/while2r
7simple_rnn_17/simple_rnn_cell_17/BiasAdd/ReadVariableOp7simple_rnn_17/simple_rnn_cell_17/BiasAdd/ReadVariableOp2p
6simple_rnn_17/simple_rnn_cell_17/MatMul/ReadVariableOp6simple_rnn_17/simple_rnn_cell_17/MatMul/ReadVariableOp2t
8simple_rnn_17/simple_rnn_cell_17/MatMul_1/ReadVariableOp8simple_rnn_17/simple_rnn_cell_17/MatMul_1/ReadVariableOp2*
simple_rnn_17/whilesimple_rnn_17/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
с
У
I__inference_sequential_12_layer_call_and_return_conditional_losses_356556

inputs&
embedding_12_356526:	'2'
simple_rnn_16_356531:	2#
simple_rnn_16_356533:	(
simple_rnn_16_356535:
(
simple_rnn_17_356538:
#
simple_rnn_17_356540:	(
simple_rnn_17_356542:
"
dense_24_356545:	@
dense_24_356547:@!
dense_25_356550:@
dense_25_356552:
identityЂ dense_24/StatefulPartitionedCallЂ dense_25/StatefulPartitionedCallЂ$embedding_12/StatefulPartitionedCallЂ%simple_rnn_16/StatefulPartitionedCallЂ%simple_rnn_17/StatefulPartitionedCallэ
$embedding_12/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_12_356526*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_12_layer_call_and_return_conditional_losses_355674\
embedding_12/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    }
embedding_12/NotEqualNotEqualinputs embedding_12/NotEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџф
%simple_rnn_16/StatefulPartitionedCallStatefulPartitionedCall-embedding_12/StatefulPartitionedCall:output:0embedding_12/NotEqual:z:0simple_rnn_16_356531simple_rnn_16_356533simple_rnn_16_356535*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_356477с
%simple_rnn_17/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_16/StatefulPartitionedCall:output:0embedding_12/NotEqual:z:0simple_rnn_17_356538simple_rnn_17_356540simple_rnn_17_356542*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_356272
 dense_24/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_17/StatefulPartitionedCall:output:0dense_24_356545dense_24_356547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_356005
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_356550dense_25_356552*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_356022x
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall%^embedding_12/StatefulPartitionedCall&^simple_rnn_16/StatefulPartitionedCall&^simple_rnn_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall2N
%simple_rnn_16/StatefulPartitionedCall%simple_rnn_16/StatefulPartitionedCall2N
%simple_rnn_17/StatefulPartitionedCall%simple_rnn_17/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	
Ц
.__inference_simple_rnn_16_layer_call_fn_357510

inputs
mask

unknown:	2
	unknown_0:	
	unknown_1:

identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsmaskunknown	unknown_0	unknown_1*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_355828t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџ2:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs:MI
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask
	
љ
while_cond_355895
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_355895___redundant_placeholder04
0while_while_cond_355895___redundant_placeholder14
0while_while_cond_355895___redundant_placeholder24
0while_while_cond_355895___redundant_placeholder34
0while_while_cond_355895___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
м
Њ
while_cond_358370
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_358370___redundant_placeholder04
0while_while_cond_358370___redundant_placeholder14
0while_while_cond_358370___redundant_placeholder24
0while_while_cond_358370___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
И
­
simple_rnn_16_while_cond_3571618
4simple_rnn_16_while_simple_rnn_16_while_loop_counter>
:simple_rnn_16_while_simple_rnn_16_while_maximum_iterations#
simple_rnn_16_while_placeholder%
!simple_rnn_16_while_placeholder_1%
!simple_rnn_16_while_placeholder_2%
!simple_rnn_16_while_placeholder_3:
6simple_rnn_16_while_less_simple_rnn_16_strided_slice_1P
Lsimple_rnn_16_while_simple_rnn_16_while_cond_357161___redundant_placeholder0P
Lsimple_rnn_16_while_simple_rnn_16_while_cond_357161___redundant_placeholder1P
Lsimple_rnn_16_while_simple_rnn_16_while_cond_357161___redundant_placeholder2P
Lsimple_rnn_16_while_simple_rnn_16_while_cond_357161___redundant_placeholder3P
Lsimple_rnn_16_while_simple_rnn_16_while_cond_357161___redundant_placeholder4 
simple_rnn_16_while_identity

simple_rnn_16/while/LessLesssimple_rnn_16_while_placeholder6simple_rnn_16_while_less_simple_rnn_16_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_16/while/IdentityIdentitysimple_rnn_16/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_16_while_identity%simple_rnn_16/while/Identity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
м
Њ
while_cond_355583
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_355583___redundant_placeholder04
0while_while_cond_355583___redundant_placeholder14
0while_while_cond_355583___redundant_placeholder24
0while_while_cond_355583___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Оq
И
simple_rnn_16_while_body_3571628
4simple_rnn_16_while_simple_rnn_16_while_loop_counter>
:simple_rnn_16_while_simple_rnn_16_while_maximum_iterations#
simple_rnn_16_while_placeholder%
!simple_rnn_16_while_placeholder_1%
!simple_rnn_16_while_placeholder_2%
!simple_rnn_16_while_placeholder_37
3simple_rnn_16_while_simple_rnn_16_strided_slice_1_0s
osimple_rnn_16_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_tensorlistfromtensor_0w
ssimple_rnn_16_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_1_tensorlistfromtensor_0Z
Gsimple_rnn_16_while_simple_rnn_cell_16_matmul_readvariableop_resource_0:	2W
Hsimple_rnn_16_while_simple_rnn_cell_16_biasadd_readvariableop_resource_0:	]
Isimple_rnn_16_while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0:
 
simple_rnn_16_while_identity"
simple_rnn_16_while_identity_1"
simple_rnn_16_while_identity_2"
simple_rnn_16_while_identity_3"
simple_rnn_16_while_identity_4"
simple_rnn_16_while_identity_55
1simple_rnn_16_while_simple_rnn_16_strided_slice_1q
msimple_rnn_16_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_tensorlistfromtensoru
qsimple_rnn_16_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_1_tensorlistfromtensorX
Esimple_rnn_16_while_simple_rnn_cell_16_matmul_readvariableop_resource:	2U
Fsimple_rnn_16_while_simple_rnn_cell_16_biasadd_readvariableop_resource:	[
Gsimple_rnn_16_while_simple_rnn_cell_16_matmul_1_readvariableop_resource:
Ђ=simple_rnn_16/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpЂ<simple_rnn_16/while/simple_rnn_cell_16/MatMul/ReadVariableOpЂ>simple_rnn_16/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp
Esimple_rnn_16/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   ь
7simple_rnn_16/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_16_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_16_while_placeholderNsimple_rnn_16/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ2*
element_dtype0
Gsimple_rnn_16/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   є
9simple_rnn_16/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemssimple_rnn_16_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_1_tensorlistfromtensor_0simple_rnn_16_while_placeholderPsimple_rnn_16/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
Є
6simple_rnn_16/while/simple_rnn_cell_16/ones_like/ShapeShape>simple_rnn_16/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:{
6simple_rnn_16/while/simple_rnn_cell_16/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ь
0simple_rnn_16/while/simple_rnn_cell_16/ones_likeFill?simple_rnn_16/while/simple_rnn_cell_16/ones_like/Shape:output:0?simple_rnn_16/while/simple_rnn_cell_16/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2y
4simple_rnn_16/while/simple_rnn_cell_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @х
2simple_rnn_16/while/simple_rnn_cell_16/dropout/MulMul9simple_rnn_16/while/simple_rnn_cell_16/ones_like:output:0=simple_rnn_16/while/simple_rnn_cell_16/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
4simple_rnn_16/while/simple_rnn_cell_16/dropout/ShapeShape9simple_rnn_16/while/simple_rnn_cell_16/ones_like:output:0*
T0*
_output_shapes
:к
Ksimple_rnn_16/while/simple_rnn_cell_16/dropout/random_uniform/RandomUniformRandomUniform=simple_rnn_16/while/simple_rnn_cell_16/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype0
=simple_rnn_16/while/simple_rnn_cell_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
;simple_rnn_16/while/simple_rnn_cell_16/dropout/GreaterEqualGreaterEqualTsimple_rnn_16/while/simple_rnn_cell_16/dropout/random_uniform/RandomUniform:output:0Fsimple_rnn_16/while/simple_rnn_cell_16/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2Н
3simple_rnn_16/while/simple_rnn_cell_16/dropout/CastCast?simple_rnn_16/while/simple_rnn_cell_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2о
4simple_rnn_16/while/simple_rnn_cell_16/dropout/Mul_1Mul6simple_rnn_16/while/simple_rnn_cell_16/dropout/Mul:z:07simple_rnn_16/while/simple_rnn_cell_16/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
8simple_rnn_16/while/simple_rnn_cell_16/ones_like_1/ShapeShape!simple_rnn_16_while_placeholder_3*
T0*
_output_shapes
:}
8simple_rnn_16/while/simple_rnn_cell_16/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ѓ
2simple_rnn_16/while/simple_rnn_cell_16/ones_like_1FillAsimple_rnn_16/while/simple_rnn_cell_16/ones_like_1/Shape:output:0Asimple_rnn_16/while/simple_rnn_cell_16/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ{
6simple_rnn_16/while/simple_rnn_cell_16/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @ь
4simple_rnn_16/while/simple_rnn_cell_16/dropout_1/MulMul;simple_rnn_16/while/simple_rnn_cell_16/ones_like_1:output:0?simple_rnn_16/while/simple_rnn_cell_16/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
6simple_rnn_16/while/simple_rnn_cell_16/dropout_1/ShapeShape;simple_rnn_16/while/simple_rnn_cell_16/ones_like_1:output:0*
T0*
_output_shapes
:п
Msimple_rnn_16/while/simple_rnn_cell_16/dropout_1/random_uniform/RandomUniformRandomUniform?simple_rnn_16/while/simple_rnn_cell_16/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0
?simple_rnn_16/while/simple_rnn_cell_16/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ђ
=simple_rnn_16/while/simple_rnn_cell_16/dropout_1/GreaterEqualGreaterEqualVsimple_rnn_16/while/simple_rnn_cell_16/dropout_1/random_uniform/RandomUniform:output:0Hsimple_rnn_16/while/simple_rnn_cell_16/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџТ
5simple_rnn_16/while/simple_rnn_cell_16/dropout_1/CastCastAsimple_rnn_16/while/simple_rnn_cell_16/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџх
6simple_rnn_16/while/simple_rnn_cell_16/dropout_1/Mul_1Mul8simple_rnn_16/while/simple_rnn_cell_16/dropout_1/Mul:z:09simple_rnn_16/while/simple_rnn_cell_16/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџн
*simple_rnn_16/while/simple_rnn_cell_16/mulMul>simple_rnn_16/while/TensorArrayV2Read/TensorListGetItem:item:08simple_rnn_16/while/simple_rnn_cell_16/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2Х
<simple_rnn_16/while/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_16_while_simple_rnn_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	2*
dtype0р
-simple_rnn_16/while/simple_rnn_cell_16/MatMulMatMul.simple_rnn_16/while/simple_rnn_cell_16/mul:z:0Dsimple_rnn_16/while/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџУ
=simple_rnn_16/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_16_while_simple_rnn_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ь
.simple_rnn_16/while/simple_rnn_cell_16/BiasAddBiasAdd7simple_rnn_16/while/simple_rnn_cell_16/MatMul:product:0Esimple_rnn_16/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ
,simple_rnn_16/while/simple_rnn_cell_16/mul_1Mul!simple_rnn_16_while_placeholder_3:simple_rnn_16/while/simple_rnn_cell_16/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЪ
>simple_rnn_16/while/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_16_while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0ц
/simple_rnn_16/while/simple_rnn_cell_16/MatMul_1MatMul0simple_rnn_16/while/simple_rnn_cell_16/mul_1:z:0Fsimple_rnn_16/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџк
*simple_rnn_16/while/simple_rnn_cell_16/addAddV27simple_rnn_16/while/simple_rnn_cell_16/BiasAdd:output:09simple_rnn_16/while/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
+simple_rnn_16/while/simple_rnn_cell_16/TanhTanh.simple_rnn_16/while/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџs
"simple_rnn_16/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      С
simple_rnn_16/while/TileTile@simple_rnn_16/while/TensorArrayV2Read_1/TensorListGetItem:item:0+simple_rnn_16/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџв
simple_rnn_16/while/SelectV2SelectV2!simple_rnn_16/while/Tile:output:0/simple_rnn_16/while/simple_rnn_cell_16/Tanh:y:0!simple_rnn_16_while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџu
$simple_rnn_16/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      Х
simple_rnn_16/while/Tile_1Tile@simple_rnn_16/while/TensorArrayV2Read_1/TensorListGetItem:item:0-simple_rnn_16/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџж
simple_rnn_16/while/SelectV2_1SelectV2#simple_rnn_16/while/Tile_1:output:0/simple_rnn_16/while/simple_rnn_cell_16/Tanh:y:0!simple_rnn_16_while_placeholder_3*
T0*(
_output_shapes
:џџџџџџџџџј
8simple_rnn_16/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_16_while_placeholder_1simple_rnn_16_while_placeholder%simple_rnn_16/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшв[
simple_rnn_16/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_16/while/addAddV2simple_rnn_16_while_placeholder"simple_rnn_16/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_16/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_16/while/add_1AddV24simple_rnn_16_while_simple_rnn_16_while_loop_counter$simple_rnn_16/while/add_1/y:output:0*
T0*
_output_shapes
: 
simple_rnn_16/while/IdentityIdentitysimple_rnn_16/while/add_1:z:0^simple_rnn_16/while/NoOp*
T0*
_output_shapes
: Ђ
simple_rnn_16/while/Identity_1Identity:simple_rnn_16_while_simple_rnn_16_while_maximum_iterations^simple_rnn_16/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_16/while/Identity_2Identitysimple_rnn_16/while/add:z:0^simple_rnn_16/while/NoOp*
T0*
_output_shapes
: А
simple_rnn_16/while/Identity_3IdentityHsimple_rnn_16/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_16/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_16/while/Identity_4Identity%simple_rnn_16/while/SelectV2:output:0^simple_rnn_16/while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџЁ
simple_rnn_16/while/Identity_5Identity'simple_rnn_16/while/SelectV2_1:output:0^simple_rnn_16/while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_16/while/NoOpNoOp>^simple_rnn_16/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp=^simple_rnn_16/while/simple_rnn_cell_16/MatMul/ReadVariableOp?^simple_rnn_16/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_16_while_identity%simple_rnn_16/while/Identity:output:0"I
simple_rnn_16_while_identity_1'simple_rnn_16/while/Identity_1:output:0"I
simple_rnn_16_while_identity_2'simple_rnn_16/while/Identity_2:output:0"I
simple_rnn_16_while_identity_3'simple_rnn_16/while/Identity_3:output:0"I
simple_rnn_16_while_identity_4'simple_rnn_16/while/Identity_4:output:0"I
simple_rnn_16_while_identity_5'simple_rnn_16/while/Identity_5:output:0"h
1simple_rnn_16_while_simple_rnn_16_strided_slice_13simple_rnn_16_while_simple_rnn_16_strided_slice_1_0"
Fsimple_rnn_16_while_simple_rnn_cell_16_biasadd_readvariableop_resourceHsimple_rnn_16_while_simple_rnn_cell_16_biasadd_readvariableop_resource_0"
Gsimple_rnn_16_while_simple_rnn_cell_16_matmul_1_readvariableop_resourceIsimple_rnn_16_while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0"
Esimple_rnn_16_while_simple_rnn_cell_16_matmul_readvariableop_resourceGsimple_rnn_16_while_simple_rnn_cell_16_matmul_readvariableop_resource_0"ш
qsimple_rnn_16_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_1_tensorlistfromtensorssimple_rnn_16_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_1_tensorlistfromtensor_0"р
msimple_rnn_16_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_tensorlistfromtensorosimple_rnn_16_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_16_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : 2~
=simple_rnn_16/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp=simple_rnn_16/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp2|
<simple_rnn_16/while/simple_rnn_cell_16/MatMul/ReadVariableOp<simple_rnn_16/while/simple_rnn_cell_16/MatMul/ReadVariableOp2
>simple_rnn_16/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp>simple_rnn_16/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ћ

-__inference_embedding_12_layer_call_fn_357466

inputs
unknown:	'2
identityЂStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_12_layer_call_and_return_conditional_losses_355674s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	
љ
while_cond_358522
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_358522___redundant_placeholder04
0while_while_cond_358522___redundant_placeholder14
0while_while_cond_358522___redundant_placeholder24
0while_while_cond_358522___redundant_placeholder34
0while_while_cond_358522___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
:


ѕ
D__inference_dense_25_layer_call_and_return_conditional_losses_356022

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
И
О
.__inference_simple_rnn_16_layer_call_fn_357498
inputs_0
unknown:	2
	unknown_0:	
	unknown_1:

identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_355320}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
"
_user_specified_name
inputs/0
	
љ
while_cond_357861
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_357861___redundant_placeholder04
0while_while_cond_357861___redundant_placeholder14
0while_while_cond_357861___redundant_placeholder24
0while_while_cond_357861___redundant_placeholder34
0while_while_cond_357861___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
Х

п
3__inference_simple_rnn_cell_16_layer_call_fn_358864

inputs
states_0
unknown:	2
	unknown_0:	
	unknown_1:

identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_355205p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџ2:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0


і
D__inference_dense_24_layer_call_and_return_conditional_losses_356005

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
РE
Р

while_body_357862
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0L
9while_simple_rnn_cell_16_matmul_readvariableop_resource_0:	2I
:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0:	O
;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorJ
7while_simple_rnn_cell_16_matmul_readvariableop_resource:	2G
8while_simple_rnn_cell_16_biasadd_readvariableop_resource:	M
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource:
Ђ/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpЂ.while/simple_rnn_cell_16/MatMul/ReadVariableOpЂ0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ2*
element_dtype0
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ў
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0

(while/simple_rnn_cell_16/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(while/simple_rnn_cell_16/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Т
"while/simple_rnn_cell_16/ones_likeFill1while/simple_rnn_cell_16/ones_like/Shape:output:01while/simple_rnn_cell_16/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2m
*while/simple_rnn_cell_16/ones_like_1/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:o
*while/simple_rnn_cell_16/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Щ
$while/simple_rnn_cell_16/ones_like_1Fill3while/simple_rnn_cell_16/ones_like_1/Shape:output:03while/simple_rnn_cell_16/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџД
while/simple_rnn_cell_16/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/simple_rnn_cell_16/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2Љ
.while/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	2*
dtype0Ж
while/simple_rnn_cell_16/MatMulMatMul while/simple_rnn_cell_16/mul:z:06while/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЇ
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Т
 while/simple_rnn_cell_16/BiasAddBiasAdd)while/simple_rnn_cell_16/MatMul:product:07while/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/simple_rnn_cell_16/mul_1Mulwhile_placeholder_3-while/simple_rnn_cell_16/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЎ
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0М
!while/simple_rnn_cell_16/MatMul_1MatMul"while/simple_rnn_cell_16/mul_1:z:08while/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџА
while/simple_rnn_cell_16/addAddV2)while/simple_rnn_cell_16/BiasAdd:output:0+while/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџz
while/simple_rnn_cell_16/TanhTanh while/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџe
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
while/SelectV2SelectV2while/Tile:output:0!while/simple_rnn_cell_16/Tanh:y:0while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџg
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
while/SelectV2_1SelectV2while/Tile_1:output:0!while/simple_rnn_cell_16/Tanh:y:0while_placeholder_3*
T0*(
_output_shapes
:џџџџџџџџџР
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: u
while/Identity_4Identitywhile/SelectV2:output:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
while/Identity_5Identitywhile/SelectV2_1:output:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџт

while/NoOpNoOp0^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_simple_rnn_cell_16_biasadd_readvariableop_resource:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_16_matmul_readvariableop_resource9while_simple_rnn_cell_16_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"А
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : 2b
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_16/MatMul/ReadVariableOp.while/simple_rnn_cell_16/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
тF
Т

while_body_358523
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0M
9while_simple_rnn_cell_17_matmul_readvariableop_resource_0:
I
:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0:	O
;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorK
7while_simple_rnn_cell_17_matmul_readvariableop_resource:
G
8while_simple_rnn_cell_17_biasadd_readvariableop_resource:	M
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource:
Ђ/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpЂ.while/simple_rnn_cell_17/MatMul/ReadVariableOpЂ0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ў
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0

(while/simple_rnn_cell_17/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(while/simple_rnn_cell_17/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?У
"while/simple_rnn_cell_17/ones_likeFill1while/simple_rnn_cell_17/ones_like/Shape:output:01while/simple_rnn_cell_17/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџm
*while/simple_rnn_cell_17/ones_like_1/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:o
*while/simple_rnn_cell_17/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Щ
$while/simple_rnn_cell_17/ones_like_1Fill3while/simple_rnn_cell_17/ones_like_1/Shape:output:03while/simple_rnn_cell_17/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЕ
while/simple_rnn_cell_17/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/simple_rnn_cell_17/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЊ
.while/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_17_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ж
while/simple_rnn_cell_17/MatMulMatMul while/simple_rnn_cell_17/mul:z:06while/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЇ
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Т
 while/simple_rnn_cell_17/BiasAddBiasAdd)while/simple_rnn_cell_17/MatMul:product:07while/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/simple_rnn_cell_17/mul_1Mulwhile_placeholder_3-while/simple_rnn_cell_17/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЎ
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0М
!while/simple_rnn_cell_17/MatMul_1MatMul"while/simple_rnn_cell_17/mul_1:z:08while/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџА
while/simple_rnn_cell_17/addAddV2)while/simple_rnn_cell_17/BiasAdd:output:0+while/simple_rnn_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџz
while/simple_rnn_cell_17/TanhTanh while/simple_rnn_cell_17/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџe
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
while/SelectV2SelectV2while/Tile:output:0!while/simple_rnn_cell_17/Tanh:y:0while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџg
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
while/SelectV2_1SelectV2while/Tile_1:output:0!while/simple_rnn_cell_17/Tanh:y:0while_placeholder_3*
T0*(
_output_shapes
:џџџџџџџџџr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ш
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: u
while/Identity_4Identitywhile/SelectV2:output:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
while/Identity_5Identitywhile/SelectV2_1:output:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџт

while/NoOpNoOp0^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_simple_rnn_cell_17_biasadd_readvariableop_resource:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_17_matmul_readvariableop_resource9while_simple_rnn_cell_17_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"А
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : 2b
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_17/MatMul/ReadVariableOp.while/simple_rnn_cell_17/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

я
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_358889

inputs
states_01
matmul_readvariableop_resource:	2.
biasadd_readvariableop_resource:	4
 matmul_1_readvariableop_resource:

identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2I
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџX
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2*
dtype0k
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ_
mul_1Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
MatMul_1MatMul	mul_1:z:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџH
TanhTanhadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџX
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџZ

Identity_1IdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџ2:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
Њ
у
!__inference__wrapped_model_355005
embedding_12_inputE
2sequential_12_embedding_12_embedding_lookup_354693:	'2`
Msequential_12_simple_rnn_16_simple_rnn_cell_16_matmul_readvariableop_resource:	2]
Nsequential_12_simple_rnn_16_simple_rnn_cell_16_biasadd_readvariableop_resource:	c
Osequential_12_simple_rnn_16_simple_rnn_cell_16_matmul_1_readvariableop_resource:
a
Msequential_12_simple_rnn_17_simple_rnn_cell_17_matmul_readvariableop_resource:
]
Nsequential_12_simple_rnn_17_simple_rnn_cell_17_biasadd_readvariableop_resource:	c
Osequential_12_simple_rnn_17_simple_rnn_cell_17_matmul_1_readvariableop_resource:
H
5sequential_12_dense_24_matmul_readvariableop_resource:	@D
6sequential_12_dense_24_biasadd_readvariableop_resource:@G
5sequential_12_dense_25_matmul_readvariableop_resource:@D
6sequential_12_dense_25_biasadd_readvariableop_resource:
identityЂ-sequential_12/dense_24/BiasAdd/ReadVariableOpЂ,sequential_12/dense_24/MatMul/ReadVariableOpЂ-sequential_12/dense_25/BiasAdd/ReadVariableOpЂ,sequential_12/dense_25/MatMul/ReadVariableOpЂ+sequential_12/embedding_12/embedding_lookupЂEsequential_12/simple_rnn_16/simple_rnn_cell_16/BiasAdd/ReadVariableOpЂDsequential_12/simple_rnn_16/simple_rnn_cell_16/MatMul/ReadVariableOpЂFsequential_12/simple_rnn_16/simple_rnn_cell_16/MatMul_1/ReadVariableOpЂ!sequential_12/simple_rnn_16/whileЂEsequential_12/simple_rnn_17/simple_rnn_cell_17/BiasAdd/ReadVariableOpЂDsequential_12/simple_rnn_17/simple_rnn_cell_17/MatMul/ReadVariableOpЂFsequential_12/simple_rnn_17/simple_rnn_cell_17/MatMul_1/ReadVariableOpЂ!sequential_12/simple_rnn_17/while|
sequential_12/embedding_12/CastCastembedding_12_input*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџЇ
+sequential_12/embedding_12/embedding_lookupResourceGather2sequential_12_embedding_12_embedding_lookup_354693#sequential_12/embedding_12/Cast:y:0*
Tindices0*E
_class;
97loc:@sequential_12/embedding_12/embedding_lookup/354693*+
_output_shapes
:џџџџџџџџџ2*
dtype0ѓ
4sequential_12/embedding_12/embedding_lookup/IdentityIdentity4sequential_12/embedding_12/embedding_lookup:output:0*
T0*E
_class;
97loc:@sequential_12/embedding_12/embedding_lookup/354693*+
_output_shapes
:џџџџџџџџџ2З
6sequential_12/embedding_12/embedding_lookup/Identity_1Identity=sequential_12/embedding_12/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2j
%sequential_12/embedding_12/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ѕ
#sequential_12/embedding_12/NotEqualNotEqualembedding_12_input.sequential_12/embedding_12/NotEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
!sequential_12/simple_rnn_16/ShapeShape?sequential_12/embedding_12/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:y
/sequential_12/simple_rnn_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_12/simple_rnn_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_12/simple_rnn_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)sequential_12/simple_rnn_16/strided_sliceStridedSlice*sequential_12/simple_rnn_16/Shape:output:08sequential_12/simple_rnn_16/strided_slice/stack:output:0:sequential_12/simple_rnn_16/strided_slice/stack_1:output:0:sequential_12/simple_rnn_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
*sequential_12/simple_rnn_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ч
(sequential_12/simple_rnn_16/zeros/packedPack2sequential_12/simple_rnn_16/strided_slice:output:03sequential_12/simple_rnn_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:l
'sequential_12/simple_rnn_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    С
!sequential_12/simple_rnn_16/zerosFill1sequential_12/simple_rnn_16/zeros/packed:output:00sequential_12/simple_rnn_16/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
*sequential_12/simple_rnn_16/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          о
%sequential_12/simple_rnn_16/transpose	Transpose?sequential_12/embedding_12/embedding_lookup/Identity_1:output:03sequential_12/simple_rnn_16/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2|
#sequential_12/simple_rnn_16/Shape_1Shape)sequential_12/simple_rnn_16/transpose:y:0*
T0*
_output_shapes
:{
1sequential_12/simple_rnn_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_12/simple_rnn_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_12/simple_rnn_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+sequential_12/simple_rnn_16/strided_slice_1StridedSlice,sequential_12/simple_rnn_16/Shape_1:output:0:sequential_12/simple_rnn_16/strided_slice_1/stack:output:0<sequential_12/simple_rnn_16/strided_slice_1/stack_1:output:0<sequential_12/simple_rnn_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*sequential_12/simple_rnn_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџШ
&sequential_12/simple_rnn_16/ExpandDims
ExpandDims'sequential_12/embedding_12/NotEqual:z:03sequential_12/simple_rnn_16/ExpandDims/dim:output:0*
T0
*+
_output_shapes
:џџџџџџџџџ
,sequential_12/simple_rnn_16/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          в
'sequential_12/simple_rnn_16/transpose_1	Transpose/sequential_12/simple_rnn_16/ExpandDims:output:05sequential_12/simple_rnn_16/transpose_1/perm:output:0*
T0
*+
_output_shapes
:џџџџџџџџџ
7sequential_12/simple_rnn_16/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
)sequential_12/simple_rnn_16/TensorArrayV2TensorListReserve@sequential_12/simple_rnn_16/TensorArrayV2/element_shape:output:04sequential_12/simple_rnn_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвЂ
Qsequential_12/simple_rnn_16/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   Д
Csequential_12/simple_rnn_16/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)sequential_12/simple_rnn_16/transpose:y:0Zsequential_12/simple_rnn_16/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв{
1sequential_12/simple_rnn_16/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_12/simple_rnn_16/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_12/simple_rnn_16/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
+sequential_12/simple_rnn_16/strided_slice_2StridedSlice)sequential_12/simple_rnn_16/transpose:y:0:sequential_12/simple_rnn_16/strided_slice_2/stack:output:0<sequential_12/simple_rnn_16/strided_slice_2/stack_1:output:0<sequential_12/simple_rnn_16/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maskЂ
>sequential_12/simple_rnn_16/simple_rnn_cell_16/ones_like/ShapeShape4sequential_12/simple_rnn_16/strided_slice_2:output:0*
T0*
_output_shapes
:
>sequential_12/simple_rnn_16/simple_rnn_cell_16/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
8sequential_12/simple_rnn_16/simple_rnn_cell_16/ones_likeFillGsequential_12/simple_rnn_16/simple_rnn_cell_16/ones_like/Shape:output:0Gsequential_12/simple_rnn_16/simple_rnn_cell_16/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
@sequential_12/simple_rnn_16/simple_rnn_cell_16/ones_like_1/ShapeShape*sequential_12/simple_rnn_16/zeros:output:0*
T0*
_output_shapes
:
@sequential_12/simple_rnn_16/simple_rnn_cell_16/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
:sequential_12/simple_rnn_16/simple_rnn_cell_16/ones_like_1FillIsequential_12/simple_rnn_16/simple_rnn_cell_16/ones_like_1/Shape:output:0Isequential_12/simple_rnn_16/simple_rnn_cell_16/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџф
2sequential_12/simple_rnn_16/simple_rnn_cell_16/mulMul4sequential_12/simple_rnn_16/strided_slice_2:output:0Asequential_12/simple_rnn_16/simple_rnn_cell_16/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2г
Dsequential_12/simple_rnn_16/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOpMsequential_12_simple_rnn_16_simple_rnn_cell_16_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype0ј
5sequential_12/simple_rnn_16/simple_rnn_cell_16/MatMulMatMul6sequential_12/simple_rnn_16/simple_rnn_cell_16/mul:z:0Lsequential_12/simple_rnn_16/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџб
Esequential_12/simple_rnn_16/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOpNsequential_12_simple_rnn_16_simple_rnn_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
6sequential_12/simple_rnn_16/simple_rnn_cell_16/BiasAddBiasAdd?sequential_12/simple_rnn_16/simple_rnn_cell_16/MatMul:product:0Msequential_12/simple_rnn_16/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџп
4sequential_12/simple_rnn_16/simple_rnn_cell_16/mul_1Mul*sequential_12/simple_rnn_16/zeros:output:0Csequential_12/simple_rnn_16/simple_rnn_cell_16/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџи
Fsequential_12/simple_rnn_16/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOpOsequential_12_simple_rnn_16_simple_rnn_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0ў
7sequential_12/simple_rnn_16/simple_rnn_cell_16/MatMul_1MatMul8sequential_12/simple_rnn_16/simple_rnn_cell_16/mul_1:z:0Nsequential_12/simple_rnn_16/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџђ
2sequential_12/simple_rnn_16/simple_rnn_cell_16/addAddV2?sequential_12/simple_rnn_16/simple_rnn_cell_16/BiasAdd:output:0Asequential_12/simple_rnn_16/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџІ
3sequential_12/simple_rnn_16/simple_rnn_cell_16/TanhTanh6sequential_12/simple_rnn_16/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
9sequential_12/simple_rnn_16/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
+sequential_12/simple_rnn_16/TensorArrayV2_1TensorListReserveBsequential_12/simple_rnn_16/TensorArrayV2_1/element_shape:output:04sequential_12/simple_rnn_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвb
 sequential_12/simple_rnn_16/timeConst*
_output_shapes
: *
dtype0*
value	B : 
9sequential_12/simple_rnn_16/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
+sequential_12/simple_rnn_16/TensorArrayV2_2TensorListReserveBsequential_12/simple_rnn_16/TensorArrayV2_2/element_shape:output:04sequential_12/simple_rnn_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШЄ
Ssequential_12/simple_rnn_16/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   К
Esequential_12/simple_rnn_16/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor+sequential_12/simple_rnn_16/transpose_1:y:0\sequential_12/simple_rnn_16/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
&sequential_12/simple_rnn_16/zeros_like	ZerosLike7sequential_12/simple_rnn_16/simple_rnn_cell_16/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
4sequential_12/simple_rnn_16/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџp
.sequential_12/simple_rnn_16/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : њ	
!sequential_12/simple_rnn_16/whileWhile7sequential_12/simple_rnn_16/while/loop_counter:output:0=sequential_12/simple_rnn_16/while/maximum_iterations:output:0)sequential_12/simple_rnn_16/time:output:04sequential_12/simple_rnn_16/TensorArrayV2_1:handle:0*sequential_12/simple_rnn_16/zeros_like:y:0*sequential_12/simple_rnn_16/zeros:output:04sequential_12/simple_rnn_16/strided_slice_1:output:0Ssequential_12/simple_rnn_16/TensorArrayUnstack/TensorListFromTensor:output_handle:0Usequential_12/simple_rnn_16/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Msequential_12_simple_rnn_16_simple_rnn_cell_16_matmul_readvariableop_resourceNsequential_12_simple_rnn_16_simple_rnn_cell_16_biasadd_readvariableop_resourceOsequential_12_simple_rnn_16_simple_rnn_cell_16_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*P
_output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *9
body1R/
-sequential_12_simple_rnn_16_while_body_354756*9
cond1R/
-sequential_12_simple_rnn_16_while_cond_354755*O
output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *
parallel_iterations 
Lsequential_12/simple_rnn_16/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
>sequential_12/simple_rnn_16/TensorArrayV2Stack/TensorListStackTensorListStack*sequential_12/simple_rnn_16/while:output:3Usequential_12/simple_rnn_16/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0
1sequential_12/simple_rnn_16/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ}
3sequential_12/simple_rnn_16/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3sequential_12/simple_rnn_16/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
+sequential_12/simple_rnn_16/strided_slice_3StridedSliceGsequential_12/simple_rnn_16/TensorArrayV2Stack/TensorListStack:tensor:0:sequential_12/simple_rnn_16/strided_slice_3/stack:output:0<sequential_12/simple_rnn_16/strided_slice_3/stack_1:output:0<sequential_12/simple_rnn_16/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
,sequential_12/simple_rnn_16/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ы
'sequential_12/simple_rnn_16/transpose_2	TransposeGsequential_12/simple_rnn_16/TensorArrayV2Stack/TensorListStack:tensor:05sequential_12/simple_rnn_16/transpose_2/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ|
!sequential_12/simple_rnn_17/ShapeShape+sequential_12/simple_rnn_16/transpose_2:y:0*
T0*
_output_shapes
:y
/sequential_12/simple_rnn_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_12/simple_rnn_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_12/simple_rnn_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)sequential_12/simple_rnn_17/strided_sliceStridedSlice*sequential_12/simple_rnn_17/Shape:output:08sequential_12/simple_rnn_17/strided_slice/stack:output:0:sequential_12/simple_rnn_17/strided_slice/stack_1:output:0:sequential_12/simple_rnn_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
*sequential_12/simple_rnn_17/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ч
(sequential_12/simple_rnn_17/zeros/packedPack2sequential_12/simple_rnn_17/strided_slice:output:03sequential_12/simple_rnn_17/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:l
'sequential_12/simple_rnn_17/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    С
!sequential_12/simple_rnn_17/zerosFill1sequential_12/simple_rnn_17/zeros/packed:output:00sequential_12/simple_rnn_17/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
*sequential_12/simple_rnn_17/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ы
%sequential_12/simple_rnn_17/transpose	Transpose+sequential_12/simple_rnn_16/transpose_2:y:03sequential_12/simple_rnn_17/transpose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ|
#sequential_12/simple_rnn_17/Shape_1Shape)sequential_12/simple_rnn_17/transpose:y:0*
T0*
_output_shapes
:{
1sequential_12/simple_rnn_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_12/simple_rnn_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_12/simple_rnn_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+sequential_12/simple_rnn_17/strided_slice_1StridedSlice,sequential_12/simple_rnn_17/Shape_1:output:0:sequential_12/simple_rnn_17/strided_slice_1/stack:output:0<sequential_12/simple_rnn_17/strided_slice_1/stack_1:output:0<sequential_12/simple_rnn_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*sequential_12/simple_rnn_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџШ
&sequential_12/simple_rnn_17/ExpandDims
ExpandDims'sequential_12/embedding_12/NotEqual:z:03sequential_12/simple_rnn_17/ExpandDims/dim:output:0*
T0
*+
_output_shapes
:џџџџџџџџџ
,sequential_12/simple_rnn_17/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          в
'sequential_12/simple_rnn_17/transpose_1	Transpose/sequential_12/simple_rnn_17/ExpandDims:output:05sequential_12/simple_rnn_17/transpose_1/perm:output:0*
T0
*+
_output_shapes
:џџџџџџџџџ
7sequential_12/simple_rnn_17/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
)sequential_12/simple_rnn_17/TensorArrayV2TensorListReserve@sequential_12/simple_rnn_17/TensorArrayV2/element_shape:output:04sequential_12/simple_rnn_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвЂ
Qsequential_12/simple_rnn_17/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Д
Csequential_12/simple_rnn_17/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)sequential_12/simple_rnn_17/transpose:y:0Zsequential_12/simple_rnn_17/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв{
1sequential_12/simple_rnn_17/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_12/simple_rnn_17/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_12/simple_rnn_17/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:і
+sequential_12/simple_rnn_17/strided_slice_2StridedSlice)sequential_12/simple_rnn_17/transpose:y:0:sequential_12/simple_rnn_17/strided_slice_2/stack:output:0<sequential_12/simple_rnn_17/strided_slice_2/stack_1:output:0<sequential_12/simple_rnn_17/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskЂ
>sequential_12/simple_rnn_17/simple_rnn_cell_17/ones_like/ShapeShape4sequential_12/simple_rnn_17/strided_slice_2:output:0*
T0*
_output_shapes
:
>sequential_12/simple_rnn_17/simple_rnn_cell_17/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
8sequential_12/simple_rnn_17/simple_rnn_cell_17/ones_likeFillGsequential_12/simple_rnn_17/simple_rnn_cell_17/ones_like/Shape:output:0Gsequential_12/simple_rnn_17/simple_rnn_cell_17/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
@sequential_12/simple_rnn_17/simple_rnn_cell_17/ones_like_1/ShapeShape*sequential_12/simple_rnn_17/zeros:output:0*
T0*
_output_shapes
:
@sequential_12/simple_rnn_17/simple_rnn_cell_17/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
:sequential_12/simple_rnn_17/simple_rnn_cell_17/ones_like_1FillIsequential_12/simple_rnn_17/simple_rnn_cell_17/ones_like_1/Shape:output:0Isequential_12/simple_rnn_17/simple_rnn_cell_17/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџх
2sequential_12/simple_rnn_17/simple_rnn_cell_17/mulMul4sequential_12/simple_rnn_17/strided_slice_2:output:0Asequential_12/simple_rnn_17/simple_rnn_cell_17/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџд
Dsequential_12/simple_rnn_17/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOpMsequential_12_simple_rnn_17_simple_rnn_cell_17_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0ј
5sequential_12/simple_rnn_17/simple_rnn_cell_17/MatMulMatMul6sequential_12/simple_rnn_17/simple_rnn_cell_17/mul:z:0Lsequential_12/simple_rnn_17/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџб
Esequential_12/simple_rnn_17/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOpNsequential_12_simple_rnn_17_simple_rnn_cell_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
6sequential_12/simple_rnn_17/simple_rnn_cell_17/BiasAddBiasAdd?sequential_12/simple_rnn_17/simple_rnn_cell_17/MatMul:product:0Msequential_12/simple_rnn_17/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџп
4sequential_12/simple_rnn_17/simple_rnn_cell_17/mul_1Mul*sequential_12/simple_rnn_17/zeros:output:0Csequential_12/simple_rnn_17/simple_rnn_cell_17/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџи
Fsequential_12/simple_rnn_17/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOpOsequential_12_simple_rnn_17_simple_rnn_cell_17_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0ў
7sequential_12/simple_rnn_17/simple_rnn_cell_17/MatMul_1MatMul8sequential_12/simple_rnn_17/simple_rnn_cell_17/mul_1:z:0Nsequential_12/simple_rnn_17/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџђ
2sequential_12/simple_rnn_17/simple_rnn_cell_17/addAddV2?sequential_12/simple_rnn_17/simple_rnn_cell_17/BiasAdd:output:0Asequential_12/simple_rnn_17/simple_rnn_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџІ
3sequential_12/simple_rnn_17/simple_rnn_cell_17/TanhTanh6sequential_12/simple_rnn_17/simple_rnn_cell_17/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
9sequential_12/simple_rnn_17/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   z
8sequential_12/simple_rnn_17/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
+sequential_12/simple_rnn_17/TensorArrayV2_1TensorListReserveBsequential_12/simple_rnn_17/TensorArrayV2_1/element_shape:output:0Asequential_12/simple_rnn_17/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвb
 sequential_12/simple_rnn_17/timeConst*
_output_shapes
: *
dtype0*
value	B : 
9sequential_12/simple_rnn_17/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
+sequential_12/simple_rnn_17/TensorArrayV2_2TensorListReserveBsequential_12/simple_rnn_17/TensorArrayV2_2/element_shape:output:04sequential_12/simple_rnn_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШЄ
Ssequential_12/simple_rnn_17/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   К
Esequential_12/simple_rnn_17/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor+sequential_12/simple_rnn_17/transpose_1:y:0\sequential_12/simple_rnn_17/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
&sequential_12/simple_rnn_17/zeros_like	ZerosLike7sequential_12/simple_rnn_17/simple_rnn_cell_17/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
4sequential_12/simple_rnn_17/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџp
.sequential_12/simple_rnn_17/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : њ	
!sequential_12/simple_rnn_17/whileWhile7sequential_12/simple_rnn_17/while/loop_counter:output:0=sequential_12/simple_rnn_17/while/maximum_iterations:output:0)sequential_12/simple_rnn_17/time:output:04sequential_12/simple_rnn_17/TensorArrayV2_1:handle:0*sequential_12/simple_rnn_17/zeros_like:y:0*sequential_12/simple_rnn_17/zeros:output:04sequential_12/simple_rnn_17/strided_slice_1:output:0Ssequential_12/simple_rnn_17/TensorArrayUnstack/TensorListFromTensor:output_handle:0Usequential_12/simple_rnn_17/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Msequential_12_simple_rnn_17_simple_rnn_cell_17_matmul_readvariableop_resourceNsequential_12_simple_rnn_17_simple_rnn_cell_17_biasadd_readvariableop_resourceOsequential_12_simple_rnn_17_simple_rnn_cell_17_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*P
_output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *9
body1R/
-sequential_12_simple_rnn_17_while_body_354901*9
cond1R/
-sequential_12_simple_rnn_17_while_cond_354900*O
output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *
parallel_iterations 
Lsequential_12/simple_rnn_17/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ћ
>sequential_12/simple_rnn_17/TensorArrayV2Stack/TensorListStackTensorListStack*sequential_12/simple_rnn_17/while:output:3Usequential_12/simple_rnn_17/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elements
1sequential_12/simple_rnn_17/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ}
3sequential_12/simple_rnn_17/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3sequential_12/simple_rnn_17/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
+sequential_12/simple_rnn_17/strided_slice_3StridedSliceGsequential_12/simple_rnn_17/TensorArrayV2Stack/TensorListStack:tensor:0:sequential_12/simple_rnn_17/strided_slice_3/stack:output:0<sequential_12/simple_rnn_17/strided_slice_3/stack_1:output:0<sequential_12/simple_rnn_17/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
,sequential_12/simple_rnn_17/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ы
'sequential_12/simple_rnn_17/transpose_2	TransposeGsequential_12/simple_rnn_17/TensorArrayV2Stack/TensorListStack:tensor:05sequential_12/simple_rnn_17/transpose_2/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџЃ
,sequential_12/dense_24/MatMul/ReadVariableOpReadVariableOp5sequential_12_dense_24_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0Х
sequential_12/dense_24/MatMulMatMul4sequential_12/simple_rnn_17/strided_slice_3:output:04sequential_12/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@ 
-sequential_12/dense_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Л
sequential_12/dense_24/BiasAddBiasAdd'sequential_12/dense_24/MatMul:product:05sequential_12/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@~
sequential_12/dense_24/ReluRelu'sequential_12/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ђ
,sequential_12/dense_25/MatMul/ReadVariableOpReadVariableOp5sequential_12_dense_25_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0К
sequential_12/dense_25/MatMulMatMul)sequential_12/dense_24/Relu:activations:04sequential_12/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
-sequential_12/dense_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
sequential_12/dense_25/BiasAddBiasAdd'sequential_12/dense_25/MatMul:product:05sequential_12/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
sequential_12/dense_25/SigmoidSigmoid'sequential_12/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџq
IdentityIdentity"sequential_12/dense_25/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЊ
NoOpNoOp.^sequential_12/dense_24/BiasAdd/ReadVariableOp-^sequential_12/dense_24/MatMul/ReadVariableOp.^sequential_12/dense_25/BiasAdd/ReadVariableOp-^sequential_12/dense_25/MatMul/ReadVariableOp,^sequential_12/embedding_12/embedding_lookupF^sequential_12/simple_rnn_16/simple_rnn_cell_16/BiasAdd/ReadVariableOpE^sequential_12/simple_rnn_16/simple_rnn_cell_16/MatMul/ReadVariableOpG^sequential_12/simple_rnn_16/simple_rnn_cell_16/MatMul_1/ReadVariableOp"^sequential_12/simple_rnn_16/whileF^sequential_12/simple_rnn_17/simple_rnn_cell_17/BiasAdd/ReadVariableOpE^sequential_12/simple_rnn_17/simple_rnn_cell_17/MatMul/ReadVariableOpG^sequential_12/simple_rnn_17/simple_rnn_cell_17/MatMul_1/ReadVariableOp"^sequential_12/simple_rnn_17/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 2^
-sequential_12/dense_24/BiasAdd/ReadVariableOp-sequential_12/dense_24/BiasAdd/ReadVariableOp2\
,sequential_12/dense_24/MatMul/ReadVariableOp,sequential_12/dense_24/MatMul/ReadVariableOp2^
-sequential_12/dense_25/BiasAdd/ReadVariableOp-sequential_12/dense_25/BiasAdd/ReadVariableOp2\
,sequential_12/dense_25/MatMul/ReadVariableOp,sequential_12/dense_25/MatMul/ReadVariableOp2Z
+sequential_12/embedding_12/embedding_lookup+sequential_12/embedding_12/embedding_lookup2
Esequential_12/simple_rnn_16/simple_rnn_cell_16/BiasAdd/ReadVariableOpEsequential_12/simple_rnn_16/simple_rnn_cell_16/BiasAdd/ReadVariableOp2
Dsequential_12/simple_rnn_16/simple_rnn_cell_16/MatMul/ReadVariableOpDsequential_12/simple_rnn_16/simple_rnn_cell_16/MatMul/ReadVariableOp2
Fsequential_12/simple_rnn_16/simple_rnn_cell_16/MatMul_1/ReadVariableOpFsequential_12/simple_rnn_16/simple_rnn_cell_16/MatMul_1/ReadVariableOp2F
!sequential_12/simple_rnn_16/while!sequential_12/simple_rnn_16/while2
Esequential_12/simple_rnn_17/simple_rnn_cell_17/BiasAdd/ReadVariableOpEsequential_12/simple_rnn_17/simple_rnn_cell_17/BiasAdd/ReadVariableOp2
Dsequential_12/simple_rnn_17/simple_rnn_cell_17/MatMul/ReadVariableOpDsequential_12/simple_rnn_17/simple_rnn_cell_17/MatMul/ReadVariableOp2
Fsequential_12/simple_rnn_17/simple_rnn_cell_17/MatMul_1/ReadVariableOpFsequential_12/simple_rnn_17/simple_rnn_cell_17/MatMul_1/ReadVariableOp2F
!sequential_12/simple_rnn_17/while!sequential_12/simple_rnn_17/while:[ W
'
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameembedding_12_input
5
Ѕ
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_355463

inputs-
simple_rnn_cell_17_355386:
(
simple_rnn_cell_17_355388:	-
simple_rnn_cell_17_355390:

identityЂ*simple_rnn_cell_17/StatefulPartitionedCallЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskю
*simple_rnn_cell_17/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_17_355386simple_rnn_cell_17_355388simple_rnn_cell_17_355390*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_355385n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_17_355386simple_rnn_cell_17_355388simple_rnn_cell_17_355390*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_355399*
condR
while_cond_355398*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   з
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ{
NoOpNoOp+^simple_rnn_cell_17/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџ: : : 2X
*simple_rnn_cell_17/StatefulPartitionedCall*simple_rnn_cell_17/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
м
Њ
while_cond_357571
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_357571___redundant_placeholder04
0while_while_cond_357571___redundant_placeholder14
0while_while_cond_357571___redundant_placeholder24
0while_while_cond_357571___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Т

)__inference_dense_25_layer_call_fn_358825

inputs
unknown:@
	unknown_0:
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_356022o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ќ
Ч
.__inference_simple_rnn_17_layer_call_fn_358166

inputs
mask

unknown:

	unknown_0:	
	unknown_1:

identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsmaskunknown	unknown_0	unknown_1*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_355986p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:џџџџџџџџџ:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:MI
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask
Ё
П
.__inference_simple_rnn_17_layer_call_fn_358154
inputs_0
unknown:

	unknown_0:	
	unknown_1:

identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_355648p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Ѕ	
І
H__inference_embedding_12_layer_call_and_return_conditional_losses_357476

inputs*
embedding_lookup_357470:	'2
identityЂembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџЛ
embedding_lookupResourceGatherembedding_lookup_357470Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/357470*+
_output_shapes
:џџџџџџџџџ2*
dtype0Ђ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/357470*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
РE
Р

while_body_355739
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0L
9while_simple_rnn_cell_16_matmul_readvariableop_resource_0:	2I
:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0:	O
;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorJ
7while_simple_rnn_cell_16_matmul_readvariableop_resource:	2G
8while_simple_rnn_cell_16_biasadd_readvariableop_resource:	M
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource:
Ђ/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpЂ.while/simple_rnn_cell_16/MatMul/ReadVariableOpЂ0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ2*
element_dtype0
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ў
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0

(while/simple_rnn_cell_16/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(while/simple_rnn_cell_16/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Т
"while/simple_rnn_cell_16/ones_likeFill1while/simple_rnn_cell_16/ones_like/Shape:output:01while/simple_rnn_cell_16/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2m
*while/simple_rnn_cell_16/ones_like_1/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:o
*while/simple_rnn_cell_16/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Щ
$while/simple_rnn_cell_16/ones_like_1Fill3while/simple_rnn_cell_16/ones_like_1/Shape:output:03while/simple_rnn_cell_16/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџД
while/simple_rnn_cell_16/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/simple_rnn_cell_16/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2Љ
.while/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	2*
dtype0Ж
while/simple_rnn_cell_16/MatMulMatMul while/simple_rnn_cell_16/mul:z:06while/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЇ
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Т
 while/simple_rnn_cell_16/BiasAddBiasAdd)while/simple_rnn_cell_16/MatMul:product:07while/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/simple_rnn_cell_16/mul_1Mulwhile_placeholder_3-while/simple_rnn_cell_16/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЎ
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0М
!while/simple_rnn_cell_16/MatMul_1MatMul"while/simple_rnn_cell_16/mul_1:z:08while/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџА
while/simple_rnn_cell_16/addAddV2)while/simple_rnn_cell_16/BiasAdd:output:0+while/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџz
while/simple_rnn_cell_16/TanhTanh while/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџe
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
while/SelectV2SelectV2while/Tile:output:0!while/simple_rnn_cell_16/Tanh:y:0while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџg
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
while/SelectV2_1SelectV2while/Tile_1:output:0!while/simple_rnn_cell_16/Tanh:y:0while_placeholder_3*
T0*(
_output_shapes
:џџџџџџџџџР
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: u
while/Identity_4Identitywhile/SelectV2:output:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
while/Identity_5Identitywhile/SelectV2_1:output:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџт

while/NoOpNoOp0^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_simple_rnn_cell_16_biasadd_readvariableop_resource:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_16_matmul_readvariableop_resource9while_simple_rnn_cell_16_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"А
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : 2b
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_16/MatMul/ReadVariableOp.while/simple_rnn_cell_16/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ъk
И
-sequential_12_simple_rnn_16_while_body_354756T
Psequential_12_simple_rnn_16_while_sequential_12_simple_rnn_16_while_loop_counterZ
Vsequential_12_simple_rnn_16_while_sequential_12_simple_rnn_16_while_maximum_iterations1
-sequential_12_simple_rnn_16_while_placeholder3
/sequential_12_simple_rnn_16_while_placeholder_13
/sequential_12_simple_rnn_16_while_placeholder_23
/sequential_12_simple_rnn_16_while_placeholder_3S
Osequential_12_simple_rnn_16_while_sequential_12_simple_rnn_16_strided_slice_1_0
sequential_12_simple_rnn_16_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_16_tensorarrayunstack_tensorlistfromtensor_0
sequential_12_simple_rnn_16_while_tensorarrayv2read_1_tensorlistgetitem_sequential_12_simple_rnn_16_tensorarrayunstack_1_tensorlistfromtensor_0h
Usequential_12_simple_rnn_16_while_simple_rnn_cell_16_matmul_readvariableop_resource_0:	2e
Vsequential_12_simple_rnn_16_while_simple_rnn_cell_16_biasadd_readvariableop_resource_0:	k
Wsequential_12_simple_rnn_16_while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0:
.
*sequential_12_simple_rnn_16_while_identity0
,sequential_12_simple_rnn_16_while_identity_10
,sequential_12_simple_rnn_16_while_identity_20
,sequential_12_simple_rnn_16_while_identity_30
,sequential_12_simple_rnn_16_while_identity_40
,sequential_12_simple_rnn_16_while_identity_5Q
Msequential_12_simple_rnn_16_while_sequential_12_simple_rnn_16_strided_slice_1
sequential_12_simple_rnn_16_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_16_tensorarrayunstack_tensorlistfromtensor
sequential_12_simple_rnn_16_while_tensorarrayv2read_1_tensorlistgetitem_sequential_12_simple_rnn_16_tensorarrayunstack_1_tensorlistfromtensorf
Ssequential_12_simple_rnn_16_while_simple_rnn_cell_16_matmul_readvariableop_resource:	2c
Tsequential_12_simple_rnn_16_while_simple_rnn_cell_16_biasadd_readvariableop_resource:	i
Usequential_12_simple_rnn_16_while_simple_rnn_cell_16_matmul_1_readvariableop_resource:
ЂKsequential_12/simple_rnn_16/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpЂJsequential_12/simple_rnn_16/while/simple_rnn_cell_16/MatMul/ReadVariableOpЂLsequential_12/simple_rnn_16/while/simple_rnn_cell_16/MatMul_1/ReadVariableOpЄ
Ssequential_12/simple_rnn_16/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   Г
Esequential_12/simple_rnn_16/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_12_simple_rnn_16_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_16_tensorarrayunstack_tensorlistfromtensor_0-sequential_12_simple_rnn_16_while_placeholder\sequential_12/simple_rnn_16/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ2*
element_dtype0І
Usequential_12/simple_rnn_16/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Л
Gsequential_12/simple_rnn_16/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemsequential_12_simple_rnn_16_while_tensorarrayv2read_1_tensorlistgetitem_sequential_12_simple_rnn_16_tensorarrayunstack_1_tensorlistfromtensor_0-sequential_12_simple_rnn_16_while_placeholder^sequential_12/simple_rnn_16/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
Р
Dsequential_12/simple_rnn_16/while/simple_rnn_cell_16/ones_like/ShapeShapeLsequential_12/simple_rnn_16/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:
Dsequential_12/simple_rnn_16/while/simple_rnn_cell_16/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
>sequential_12/simple_rnn_16/while/simple_rnn_cell_16/ones_likeFillMsequential_12/simple_rnn_16/while/simple_rnn_cell_16/ones_like/Shape:output:0Msequential_12/simple_rnn_16/while/simple_rnn_cell_16/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ѕ
Fsequential_12/simple_rnn_16/while/simple_rnn_cell_16/ones_like_1/ShapeShape/sequential_12_simple_rnn_16_while_placeholder_3*
T0*
_output_shapes
:
Fsequential_12/simple_rnn_16/while/simple_rnn_cell_16/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
@sequential_12/simple_rnn_16/while/simple_rnn_cell_16/ones_like_1FillOsequential_12/simple_rnn_16/while/simple_rnn_cell_16/ones_like_1/Shape:output:0Osequential_12/simple_rnn_16/while/simple_rnn_cell_16/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
8sequential_12/simple_rnn_16/while/simple_rnn_cell_16/mulMulLsequential_12/simple_rnn_16/while/TensorArrayV2Read/TensorListGetItem:item:0Gsequential_12/simple_rnn_16/while/simple_rnn_cell_16/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2с
Jsequential_12/simple_rnn_16/while/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOpUsequential_12_simple_rnn_16_while_simple_rnn_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	2*
dtype0
;sequential_12/simple_rnn_16/while/simple_rnn_cell_16/MatMulMatMul<sequential_12/simple_rnn_16/while/simple_rnn_cell_16/mul:z:0Rsequential_12/simple_rnn_16/while/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџп
Ksequential_12/simple_rnn_16/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOpVsequential_12_simple_rnn_16_while_simple_rnn_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0
<sequential_12/simple_rnn_16/while/simple_rnn_cell_16/BiasAddBiasAddEsequential_12/simple_rnn_16/while/simple_rnn_cell_16/MatMul:product:0Ssequential_12/simple_rnn_16/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ№
:sequential_12/simple_rnn_16/while/simple_rnn_cell_16/mul_1Mul/sequential_12_simple_rnn_16_while_placeholder_3Isequential_12/simple_rnn_16/while/simple_rnn_cell_16/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџц
Lsequential_12/simple_rnn_16/while/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOpWsequential_12_simple_rnn_16_while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
=sequential_12/simple_rnn_16/while/simple_rnn_cell_16/MatMul_1MatMul>sequential_12/simple_rnn_16/while/simple_rnn_cell_16/mul_1:z:0Tsequential_12/simple_rnn_16/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
8sequential_12/simple_rnn_16/while/simple_rnn_cell_16/addAddV2Esequential_12/simple_rnn_16/while/simple_rnn_cell_16/BiasAdd:output:0Gsequential_12/simple_rnn_16/while/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџВ
9sequential_12/simple_rnn_16/while/simple_rnn_cell_16/TanhTanh<sequential_12/simple_rnn_16/while/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
0sequential_12/simple_rnn_16/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ы
&sequential_12/simple_rnn_16/while/TileTileNsequential_12/simple_rnn_16/while/TensorArrayV2Read_1/TensorListGetItem:item:09sequential_12/simple_rnn_16/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
*sequential_12/simple_rnn_16/while/SelectV2SelectV2/sequential_12/simple_rnn_16/while/Tile:output:0=sequential_12/simple_rnn_16/while/simple_rnn_cell_16/Tanh:y:0/sequential_12_simple_rnn_16_while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџ
2sequential_12/simple_rnn_16/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      я
(sequential_12/simple_rnn_16/while/Tile_1TileNsequential_12/simple_rnn_16/while/TensorArrayV2Read_1/TensorListGetItem:item:0;sequential_12/simple_rnn_16/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
,sequential_12/simple_rnn_16/while/SelectV2_1SelectV21sequential_12/simple_rnn_16/while/Tile_1:output:0=sequential_12/simple_rnn_16/while/simple_rnn_cell_16/Tanh:y:0/sequential_12_simple_rnn_16_while_placeholder_3*
T0*(
_output_shapes
:џџџџџџџџџА
Fsequential_12/simple_rnn_16/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem/sequential_12_simple_rnn_16_while_placeholder_1-sequential_12_simple_rnn_16_while_placeholder3sequential_12/simple_rnn_16/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшвi
'sequential_12/simple_rnn_16/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :А
%sequential_12/simple_rnn_16/while/addAddV2-sequential_12_simple_rnn_16_while_placeholder0sequential_12/simple_rnn_16/while/add/y:output:0*
T0*
_output_shapes
: k
)sequential_12/simple_rnn_16/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :з
'sequential_12/simple_rnn_16/while/add_1AddV2Psequential_12_simple_rnn_16_while_sequential_12_simple_rnn_16_while_loop_counter2sequential_12/simple_rnn_16/while/add_1/y:output:0*
T0*
_output_shapes
: ­
*sequential_12/simple_rnn_16/while/IdentityIdentity+sequential_12/simple_rnn_16/while/add_1:z:0'^sequential_12/simple_rnn_16/while/NoOp*
T0*
_output_shapes
: к
,sequential_12/simple_rnn_16/while/Identity_1IdentityVsequential_12_simple_rnn_16_while_sequential_12_simple_rnn_16_while_maximum_iterations'^sequential_12/simple_rnn_16/while/NoOp*
T0*
_output_shapes
: ­
,sequential_12/simple_rnn_16/while/Identity_2Identity)sequential_12/simple_rnn_16/while/add:z:0'^sequential_12/simple_rnn_16/while/NoOp*
T0*
_output_shapes
: к
,sequential_12/simple_rnn_16/while/Identity_3IdentityVsequential_12/simple_rnn_16/while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^sequential_12/simple_rnn_16/while/NoOp*
T0*
_output_shapes
: Щ
,sequential_12/simple_rnn_16/while/Identity_4Identity3sequential_12/simple_rnn_16/while/SelectV2:output:0'^sequential_12/simple_rnn_16/while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџЫ
,sequential_12/simple_rnn_16/while/Identity_5Identity5sequential_12/simple_rnn_16/while/SelectV2_1:output:0'^sequential_12/simple_rnn_16/while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџв
&sequential_12/simple_rnn_16/while/NoOpNoOpL^sequential_12/simple_rnn_16/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpK^sequential_12/simple_rnn_16/while/simple_rnn_cell_16/MatMul/ReadVariableOpM^sequential_12/simple_rnn_16/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "a
*sequential_12_simple_rnn_16_while_identity3sequential_12/simple_rnn_16/while/Identity:output:0"e
,sequential_12_simple_rnn_16_while_identity_15sequential_12/simple_rnn_16/while/Identity_1:output:0"e
,sequential_12_simple_rnn_16_while_identity_25sequential_12/simple_rnn_16/while/Identity_2:output:0"e
,sequential_12_simple_rnn_16_while_identity_35sequential_12/simple_rnn_16/while/Identity_3:output:0"e
,sequential_12_simple_rnn_16_while_identity_45sequential_12/simple_rnn_16/while/Identity_4:output:0"e
,sequential_12_simple_rnn_16_while_identity_55sequential_12/simple_rnn_16/while/Identity_5:output:0" 
Msequential_12_simple_rnn_16_while_sequential_12_simple_rnn_16_strided_slice_1Osequential_12_simple_rnn_16_while_sequential_12_simple_rnn_16_strided_slice_1_0"Ў
Tsequential_12_simple_rnn_16_while_simple_rnn_cell_16_biasadd_readvariableop_resourceVsequential_12_simple_rnn_16_while_simple_rnn_cell_16_biasadd_readvariableop_resource_0"А
Usequential_12_simple_rnn_16_while_simple_rnn_cell_16_matmul_1_readvariableop_resourceWsequential_12_simple_rnn_16_while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0"Ќ
Ssequential_12_simple_rnn_16_while_simple_rnn_cell_16_matmul_readvariableop_resourceUsequential_12_simple_rnn_16_while_simple_rnn_cell_16_matmul_readvariableop_resource_0"Ђ
sequential_12_simple_rnn_16_while_tensorarrayv2read_1_tensorlistgetitem_sequential_12_simple_rnn_16_tensorarrayunstack_1_tensorlistfromtensorsequential_12_simple_rnn_16_while_tensorarrayv2read_1_tensorlistgetitem_sequential_12_simple_rnn_16_tensorarrayunstack_1_tensorlistfromtensor_0"
sequential_12_simple_rnn_16_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_16_tensorarrayunstack_tensorlistfromtensorsequential_12_simple_rnn_16_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_16_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : 2
Ksequential_12/simple_rnn_16/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpKsequential_12/simple_rnn_16/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp2
Jsequential_12/simple_rnn_16/while/simple_rnn_cell_16/MatMul/ReadVariableOpJsequential_12/simple_rnn_16/while/simple_rnn_cell_16/MatMul/ReadVariableOp2
Lsequential_12/simple_rnn_16/while/simple_rnn_cell_16/MatMul_1/ReadVariableOpLsequential_12/simple_rnn_16/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
5
Ѕ
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_355648

inputs-
simple_rnn_cell_17_355571:
(
simple_rnn_cell_17_355573:	-
simple_rnn_cell_17_355575:

identityЂ*simple_rnn_cell_17/StatefulPartitionedCallЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskю
*simple_rnn_cell_17/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_17_355571simple_rnn_cell_17_355573simple_rnn_cell_17_355575*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_355531n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_17_355571simple_rnn_cell_17_355573simple_rnn_cell_17_355575*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_355584*
condR
while_cond_355583*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   з
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ{
NoOpNoOp+^simple_rnn_cell_17/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџ: : : 2X
*simple_rnn_cell_17/StatefulPartitionedCall*simple_rnn_cell_17/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
f
б
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_356272

inputs
mask
E
1simple_rnn_cell_17_matmul_readvariableop_resource:
A
2simple_rnn_cell_17_biasadd_readvariableop_resource:	G
3simple_rnn_cell_17_matmul_1_readvariableop_resource:

identityЂ)simple_rnn_cell_17/BiasAdd/ReadVariableOpЂ(simple_rnn_cell_17/MatMul/ReadVariableOpЂ*simple_rnn_cell_17/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџm

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*+
_output_shapes
:џџџџџџџџџe
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ~
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*+
_output_shapes
:џџџџџџџџџf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskj
"simple_rnn_cell_17/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:g
"simple_rnn_cell_17/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Б
simple_rnn_cell_17/ones_likeFill+simple_rnn_cell_17/ones_like/Shape:output:0+simple_rnn_cell_17/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџe
 simple_rnn_cell_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Њ
simple_rnn_cell_17/dropout/MulMul%simple_rnn_cell_17/ones_like:output:0)simple_rnn_cell_17/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџu
 simple_rnn_cell_17/dropout/ShapeShape%simple_rnn_cell_17/ones_like:output:0*
T0*
_output_shapes
:Г
7simple_rnn_cell_17/dropout/random_uniform/RandomUniformRandomUniform)simple_rnn_cell_17/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0n
)simple_rnn_cell_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?р
'simple_rnn_cell_17/dropout/GreaterEqualGreaterEqual@simple_rnn_cell_17/dropout/random_uniform/RandomUniform:output:02simple_rnn_cell_17/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/dropout/CastCast+simple_rnn_cell_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЃ
 simple_rnn_cell_17/dropout/Mul_1Mul"simple_rnn_cell_17/dropout/Mul:z:0#simple_rnn_cell_17/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџb
$simple_rnn_cell_17/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:i
$simple_rnn_cell_17/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?З
simple_rnn_cell_17/ones_like_1Fill-simple_rnn_cell_17/ones_like_1/Shape:output:0-simple_rnn_cell_17/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџg
"simple_rnn_cell_17/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @А
 simple_rnn_cell_17/dropout_1/MulMul'simple_rnn_cell_17/ones_like_1:output:0+simple_rnn_cell_17/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџy
"simple_rnn_cell_17/dropout_1/ShapeShape'simple_rnn_cell_17/ones_like_1:output:0*
T0*
_output_shapes
:З
9simple_rnn_cell_17/dropout_1/random_uniform/RandomUniformRandomUniform+simple_rnn_cell_17/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0p
+simple_rnn_cell_17/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ц
)simple_rnn_cell_17/dropout_1/GreaterEqualGreaterEqualBsimple_rnn_cell_17/dropout_1/random_uniform/RandomUniform:output:04simple_rnn_cell_17/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
!simple_rnn_cell_17/dropout_1/CastCast-simple_rnn_cell_17/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЉ
"simple_rnn_cell_17/dropout_1/Mul_1Mul$simple_rnn_cell_17/dropout_1/Mul:z:0%simple_rnn_cell_17/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/mulMulstrided_slice_2:output:0$simple_rnn_cell_17/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
(simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_17_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Є
simple_rnn_cell_17/MatMulMatMulsimple_rnn_cell_17/mul:z:00simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
)simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0А
simple_rnn_cell_17/BiasAddBiasAdd#simple_rnn_cell_17/MatMul:product:01simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/mul_1Mulzeros:output:0&simple_rnn_cell_17/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ 
*simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_17_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Њ
simple_rnn_cell_17/MatMul_1MatMulsimple_rnn_cell_17/mul_1:z:02simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/addAddV2#simple_rnn_cell_17/BiasAdd:output:0%simple_rnn_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџn
simple_rnn_cell_17/TanhTanhsimple_rnn_cell_17/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : h
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџИ
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ц
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШg

zeros_like	ZerosLikesimple_rnn_cell_17/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџc
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ж
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:01simple_rnn_cell_17_matmul_readvariableop_resource2simple_rnn_cell_17_biasadd_readvariableop_resource3simple_rnn_cell_17_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*P
_output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_356166*
condR
while_cond_356165*O
output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   з
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџв
NoOpNoOp*^simple_rnn_cell_17/BiasAdd/ReadVariableOp)^simple_rnn_cell_17/MatMul/ReadVariableOp+^simple_rnn_cell_17/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:џџџџџџџџџ:џџџџџџџџџ: : : 2V
)simple_rnn_cell_17/BiasAdd/ReadVariableOp)simple_rnn_cell_17/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_17/MatMul/ReadVariableOp(simple_rnn_cell_17/MatMul/ReadVariableOp2X
*simple_rnn_cell_17/MatMul_1/ReadVariableOp*simple_rnn_cell_17/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:MI
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask
У$
ю
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_355531

inputs

states2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	4
 matmul_1_readvariableop_resource:

identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @q
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџO
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџG
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @w
dropout_1/MulMulones_like_1:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџS
dropout_1/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџt
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџp
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџX
mulMulinputsdropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0k
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ\
mul_1Mulstatesdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
MatMul_1MatMul	mul_1:z:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџH
TanhTanhadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџX
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџZ

Identity_1IdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:PL
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates
П\

__inference__traced_save_359173
file_prefix6
2savev2_embedding_12_embeddings_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableopF
Bsavev2_simple_rnn_16_simple_rnn_cell_16_kernel_read_readvariableopP
Lsavev2_simple_rnn_16_simple_rnn_cell_16_recurrent_kernel_read_readvariableopD
@savev2_simple_rnn_16_simple_rnn_cell_16_bias_read_readvariableopF
Bsavev2_simple_rnn_17_simple_rnn_cell_17_kernel_read_readvariableopP
Lsavev2_simple_rnn_17_simple_rnn_cell_17_recurrent_kernel_read_readvariableopD
@savev2_simple_rnn_17_simple_rnn_cell_17_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop=
9savev2_adam_embedding_12_embeddings_m_read_readvariableop5
1savev2_adam_dense_24_kernel_m_read_readvariableop3
/savev2_adam_dense_24_bias_m_read_readvariableop5
1savev2_adam_dense_25_kernel_m_read_readvariableop3
/savev2_adam_dense_25_bias_m_read_readvariableopM
Isavev2_adam_simple_rnn_16_simple_rnn_cell_16_kernel_m_read_readvariableopW
Ssavev2_adam_simple_rnn_16_simple_rnn_cell_16_recurrent_kernel_m_read_readvariableopK
Gsavev2_adam_simple_rnn_16_simple_rnn_cell_16_bias_m_read_readvariableopM
Isavev2_adam_simple_rnn_17_simple_rnn_cell_17_kernel_m_read_readvariableopW
Ssavev2_adam_simple_rnn_17_simple_rnn_cell_17_recurrent_kernel_m_read_readvariableopK
Gsavev2_adam_simple_rnn_17_simple_rnn_cell_17_bias_m_read_readvariableop=
9savev2_adam_embedding_12_embeddings_v_read_readvariableop5
1savev2_adam_dense_24_kernel_v_read_readvariableop3
/savev2_adam_dense_24_bias_v_read_readvariableop5
1savev2_adam_dense_25_kernel_v_read_readvariableop3
/savev2_adam_dense_25_bias_v_read_readvariableopM
Isavev2_adam_simple_rnn_16_simple_rnn_cell_16_kernel_v_read_readvariableopW
Ssavev2_adam_simple_rnn_16_simple_rnn_cell_16_recurrent_kernel_v_read_readvariableopK
Gsavev2_adam_simple_rnn_16_simple_rnn_cell_16_bias_v_read_readvariableopM
Isavev2_adam_simple_rnn_17_simple_rnn_cell_17_kernel_v_read_readvariableopW
Ssavev2_adam_simple_rnn_17_simple_rnn_cell_17_recurrent_kernel_v_read_readvariableopK
Gsavev2_adam_simple_rnn_17_simple_rnn_cell_17_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ч
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*№
valueцBу+B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHУ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ф
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_embedding_12_embeddings_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableopBsavev2_simple_rnn_16_simple_rnn_cell_16_kernel_read_readvariableopLsavev2_simple_rnn_16_simple_rnn_cell_16_recurrent_kernel_read_readvariableop@savev2_simple_rnn_16_simple_rnn_cell_16_bias_read_readvariableopBsavev2_simple_rnn_17_simple_rnn_cell_17_kernel_read_readvariableopLsavev2_simple_rnn_17_simple_rnn_cell_17_recurrent_kernel_read_readvariableop@savev2_simple_rnn_17_simple_rnn_cell_17_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop9savev2_adam_embedding_12_embeddings_m_read_readvariableop1savev2_adam_dense_24_kernel_m_read_readvariableop/savev2_adam_dense_24_bias_m_read_readvariableop1savev2_adam_dense_25_kernel_m_read_readvariableop/savev2_adam_dense_25_bias_m_read_readvariableopIsavev2_adam_simple_rnn_16_simple_rnn_cell_16_kernel_m_read_readvariableopSsavev2_adam_simple_rnn_16_simple_rnn_cell_16_recurrent_kernel_m_read_readvariableopGsavev2_adam_simple_rnn_16_simple_rnn_cell_16_bias_m_read_readvariableopIsavev2_adam_simple_rnn_17_simple_rnn_cell_17_kernel_m_read_readvariableopSsavev2_adam_simple_rnn_17_simple_rnn_cell_17_recurrent_kernel_m_read_readvariableopGsavev2_adam_simple_rnn_17_simple_rnn_cell_17_bias_m_read_readvariableop9savev2_adam_embedding_12_embeddings_v_read_readvariableop1savev2_adam_dense_24_kernel_v_read_readvariableop/savev2_adam_dense_24_bias_v_read_readvariableop1savev2_adam_dense_25_kernel_v_read_readvariableop/savev2_adam_dense_25_bias_v_read_readvariableopIsavev2_adam_simple_rnn_16_simple_rnn_cell_16_kernel_v_read_readvariableopSsavev2_adam_simple_rnn_16_simple_rnn_cell_16_recurrent_kernel_v_read_readvariableopGsavev2_adam_simple_rnn_16_simple_rnn_cell_16_bias_v_read_readvariableopIsavev2_adam_simple_rnn_17_simple_rnn_cell_17_kernel_v_read_readvariableopSsavev2_adam_simple_rnn_17_simple_rnn_cell_17_recurrent_kernel_v_read_readvariableopGsavev2_adam_simple_rnn_17_simple_rnn_cell_17_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ц
_input_shapesд
б: :	'2:	@:@:@::	2:
::
:
:: : : : : : : : : :	'2:	@:@:@::	2:
::
:
::	'2:	@:@:@::	2:
::
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	'2:%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	2:&"
 
_output_shapes
:
:!

_output_shapes	
::&	"
 
_output_shapes
:
:&
"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	'2:%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	2:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::% !

_output_shapes
:	'2:%!!

_output_shapes
:	@: "

_output_shapes
:@:$# 

_output_shapes

:@: $

_output_shapes
::%%!

_output_shapes
:	2:&&"
 
_output_shapes
:
:!'

_output_shapes	
::&("
 
_output_shapes
:
:&)"
 
_output_shapes
:
:!*

_output_shapes	
::+

_output_shapes
: 
їd
а
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_356477

inputs
mask
D
1simple_rnn_cell_16_matmul_readvariableop_resource:	2A
2simple_rnn_cell_16_biasadd_readvariableop_resource:	G
3simple_rnn_cell_16_matmul_1_readvariableop_resource:

identityЂ)simple_rnn_cell_16/BiasAdd/ReadVariableOpЂ(simple_rnn_cell_16/MatMul/ReadVariableOpЂ*simple_rnn_cell_16/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџm

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*+
_output_shapes
:џџџџџџџџџe
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ~
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*+
_output_shapes
:џџџџџџџџџf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maskj
"simple_rnn_cell_16/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:g
"simple_rnn_cell_16/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?А
simple_rnn_cell_16/ones_likeFill+simple_rnn_cell_16/ones_like/Shape:output:0+simple_rnn_cell_16/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2e
 simple_rnn_cell_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Љ
simple_rnn_cell_16/dropout/MulMul%simple_rnn_cell_16/ones_like:output:0)simple_rnn_cell_16/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2u
 simple_rnn_cell_16/dropout/ShapeShape%simple_rnn_cell_16/ones_like:output:0*
T0*
_output_shapes
:В
7simple_rnn_cell_16/dropout/random_uniform/RandomUniformRandomUniform)simple_rnn_cell_16/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype0n
)simple_rnn_cell_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?п
'simple_rnn_cell_16/dropout/GreaterEqualGreaterEqual@simple_rnn_cell_16/dropout/random_uniform/RandomUniform:output:02simple_rnn_cell_16/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_16/dropout/CastCast+simple_rnn_cell_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2Ђ
 simple_rnn_cell_16/dropout/Mul_1Mul"simple_rnn_cell_16/dropout/Mul:z:0#simple_rnn_cell_16/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2b
$simple_rnn_cell_16/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:i
$simple_rnn_cell_16/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?З
simple_rnn_cell_16/ones_like_1Fill-simple_rnn_cell_16/ones_like_1/Shape:output:0-simple_rnn_cell_16/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџg
"simple_rnn_cell_16/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @А
 simple_rnn_cell_16/dropout_1/MulMul'simple_rnn_cell_16/ones_like_1:output:0+simple_rnn_cell_16/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџy
"simple_rnn_cell_16/dropout_1/ShapeShape'simple_rnn_cell_16/ones_like_1:output:0*
T0*
_output_shapes
:З
9simple_rnn_cell_16/dropout_1/random_uniform/RandomUniformRandomUniform+simple_rnn_cell_16/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0p
+simple_rnn_cell_16/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ц
)simple_rnn_cell_16/dropout_1/GreaterEqualGreaterEqualBsimple_rnn_cell_16/dropout_1/random_uniform/RandomUniform:output:04simple_rnn_cell_16/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
!simple_rnn_cell_16/dropout_1/CastCast-simple_rnn_cell_16/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЉ
"simple_rnn_cell_16/dropout_1/Mul_1Mul$simple_rnn_cell_16/dropout_1/Mul:z:0%simple_rnn_cell_16/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_16/mulMulstrided_slice_2:output:0$simple_rnn_cell_16/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
(simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_16_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype0Є
simple_rnn_cell_16/MatMulMatMulsimple_rnn_cell_16/mul:z:00simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
)simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0А
simple_rnn_cell_16/BiasAddBiasAdd#simple_rnn_cell_16/MatMul:product:01simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_16/mul_1Mulzeros:output:0&simple_rnn_cell_16/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ 
*simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Њ
simple_rnn_cell_16/MatMul_1MatMulsimple_rnn_cell_16/mul_1:z:02simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_16/addAddV2#simple_rnn_cell_16/BiasAdd:output:0%simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџn
simple_rnn_cell_16/TanhTanhsimple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : h
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџИ
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ц
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШg

zeros_like	ZerosLikesimple_rnn_cell_16/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџc
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ж
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:01simple_rnn_cell_16_matmul_readvariableop_resource2simple_rnn_cell_16_biasadd_readvariableop_resource3simple_rnn_cell_16_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*P
_output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_356372*
condR
while_cond_356371*O
output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   У
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџc
IdentityIdentitytranspose_2:y:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџв
NoOpNoOp*^simple_rnn_cell_16/BiasAdd/ReadVariableOp)^simple_rnn_cell_16/MatMul/ReadVariableOp+^simple_rnn_cell_16/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџ2:џџџџџџџџџ: : : 2V
)simple_rnn_cell_16/BiasAdd/ReadVariableOp)simple_rnn_cell_16/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_16/MatMul/ReadVariableOp(simple_rnn_cell_16/MatMul/ReadVariableOp2X
*simple_rnn_cell_16/MatMul_1/ReadVariableOp*simple_rnn_cell_16/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs:MI
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask
Ё
П
.__inference_simple_rnn_17_layer_call_fn_358143
inputs_0
unknown:

	unknown_0:	
	unknown_1:

identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_355463p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
ћ

Ћ
.__inference_sequential_12_layer_call_fn_356054
embedding_12_input
unknown:	'2
	unknown_0:	2
	unknown_1:	
	unknown_2:

	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	@
	unknown_7:@
	unknown_8:@
	unknown_9:
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallembedding_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_356029o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameembedding_12_input
	
Ц
.__inference_simple_rnn_16_layer_call_fn_357522

inputs
mask

unknown:	2
	unknown_0:	
	unknown_1:

identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsmaskunknown	unknown_0	unknown_1*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_356477t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџ2:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs:MI
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask

э
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_355061

inputs

states1
matmul_readvariableop_resource:	2.
biasadd_readvariableop_resource:	4
 matmul_1_readvariableop_resource:

identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2G
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџX
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2*
dtype0k
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ]
mul_1Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
MatMul_1MatMul	mul_1:z:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџH
TanhTanhadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџX
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџZ

Identity_1IdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџ2:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs:PL
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates
	
љ
while_cond_358026
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_358026___redundant_placeholder04
0while_while_cond_358026___redundant_placeholder14
0while_while_cond_358026___redundant_placeholder24
0while_while_cond_358026___redundant_placeholder34
0while_while_cond_358026___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
	
љ
while_cond_356165
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_356165___redundant_placeholder04
0while_while_cond_356165___redundant_placeholder14
0while_while_cond_356165___redundant_placeholder24
0while_while_cond_356165___redundant_placeholder34
0while_while_cond_356165___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
м
Њ
while_cond_355398
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_355398___redundant_placeholder04
0while_while_cond_355398___redundant_placeholder14
0while_while_cond_355398___redundant_placeholder24
0while_while_cond_355398___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Ы$
№
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_359024

inputs
states_02
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	4
 matmul_1_readvariableop_resource:

identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @q
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџO
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџI
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @w
dropout_1/MulMulones_like_1:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџS
dropout_1/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџt
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџp
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџX
mulMulinputsdropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0k
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ^
mul_1Mulstates_0dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
MatMul_1MatMul	mul_1:z:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџH
TanhTanhadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџX
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџZ

Identity_1IdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
ЙK
й
while_body_357712
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_16_matmul_readvariableop_resource_0:	2I
:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0:	O
;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_16_matmul_readvariableop_resource:	2G
8while_simple_rnn_cell_16_biasadd_readvariableop_resource:	M
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource:
Ђ/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpЂ.while/simple_rnn_cell_16/MatMul/ReadVariableOpЂ0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ2*
element_dtype0
(while/simple_rnn_cell_16/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(while/simple_rnn_cell_16/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Т
"while/simple_rnn_cell_16/ones_likeFill1while/simple_rnn_cell_16/ones_like/Shape:output:01while/simple_rnn_cell_16/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2k
&while/simple_rnn_cell_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Л
$while/simple_rnn_cell_16/dropout/MulMul+while/simple_rnn_cell_16/ones_like:output:0/while/simple_rnn_cell_16/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
&while/simple_rnn_cell_16/dropout/ShapeShape+while/simple_rnn_cell_16/ones_like:output:0*
T0*
_output_shapes
:О
=while/simple_rnn_cell_16/dropout/random_uniform/RandomUniformRandomUniform/while/simple_rnn_cell_16/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype0t
/while/simple_rnn_cell_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ё
-while/simple_rnn_cell_16/dropout/GreaterEqualGreaterEqualFwhile/simple_rnn_cell_16/dropout/random_uniform/RandomUniform:output:08while/simple_rnn_cell_16/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ё
%while/simple_rnn_cell_16/dropout/CastCast1while/simple_rnn_cell_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2Д
&while/simple_rnn_cell_16/dropout/Mul_1Mul(while/simple_rnn_cell_16/dropout/Mul:z:0)while/simple_rnn_cell_16/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2m
*while/simple_rnn_cell_16/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:o
*while/simple_rnn_cell_16/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Щ
$while/simple_rnn_cell_16/ones_like_1Fill3while/simple_rnn_cell_16/ones_like_1/Shape:output:03while/simple_rnn_cell_16/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџm
(while/simple_rnn_cell_16/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Т
&while/simple_rnn_cell_16/dropout_1/MulMul-while/simple_rnn_cell_16/ones_like_1:output:01while/simple_rnn_cell_16/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
(while/simple_rnn_cell_16/dropout_1/ShapeShape-while/simple_rnn_cell_16/ones_like_1:output:0*
T0*
_output_shapes
:У
?while/simple_rnn_cell_16/dropout_1/random_uniform/RandomUniformRandomUniform1while/simple_rnn_cell_16/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0v
1while/simple_rnn_cell_16/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ј
/while/simple_rnn_cell_16/dropout_1/GreaterEqualGreaterEqualHwhile/simple_rnn_cell_16/dropout_1/random_uniform/RandomUniform:output:0:while/simple_rnn_cell_16/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџІ
'while/simple_rnn_cell_16/dropout_1/CastCast3while/simple_rnn_cell_16/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЛ
(while/simple_rnn_cell_16/dropout_1/Mul_1Mul*while/simple_rnn_cell_16/dropout_1/Mul:z:0+while/simple_rnn_cell_16/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџГ
while/simple_rnn_cell_16/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/simple_rnn_cell_16/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2Љ
.while/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	2*
dtype0Ж
while/simple_rnn_cell_16/MatMulMatMul while/simple_rnn_cell_16/mul:z:06while/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЇ
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Т
 while/simple_rnn_cell_16/BiasAddBiasAdd)while/simple_rnn_cell_16/MatMul:product:07while/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/simple_rnn_cell_16/mul_1Mulwhile_placeholder_2,while/simple_rnn_cell_16/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЎ
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0М
!while/simple_rnn_cell_16/MatMul_1MatMul"while/simple_rnn_cell_16/mul_1:z:08while/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџА
while/simple_rnn_cell_16/addAddV2)while/simple_rnn_cell_16/BiasAdd:output:0+while/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџz
while/simple_rnn_cell_16/TanhTanh while/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЪ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_16/Tanh:y:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity!while/simple_rnn_cell_16/Tanh:y:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџт

while/NoOpNoOp0^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_16_biasadd_readvariableop_resource:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_16_matmul_readvariableop_resource9while_simple_rnn_cell_16_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :џџџџџџџџџ: : : : : 2b
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_16/MatMul/ReadVariableOp.while/simple_rnn_cell_16/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
ЯZ
Р

while_body_356372
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0L
9while_simple_rnn_cell_16_matmul_readvariableop_resource_0:	2I
:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0:	O
;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorJ
7while_simple_rnn_cell_16_matmul_readvariableop_resource:	2G
8while_simple_rnn_cell_16_biasadd_readvariableop_resource:	M
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource:
Ђ/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpЂ.while/simple_rnn_cell_16/MatMul/ReadVariableOpЂ0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ2*
element_dtype0
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ў
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0

(while/simple_rnn_cell_16/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(while/simple_rnn_cell_16/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Т
"while/simple_rnn_cell_16/ones_likeFill1while/simple_rnn_cell_16/ones_like/Shape:output:01while/simple_rnn_cell_16/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2k
&while/simple_rnn_cell_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Л
$while/simple_rnn_cell_16/dropout/MulMul+while/simple_rnn_cell_16/ones_like:output:0/while/simple_rnn_cell_16/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
&while/simple_rnn_cell_16/dropout/ShapeShape+while/simple_rnn_cell_16/ones_like:output:0*
T0*
_output_shapes
:О
=while/simple_rnn_cell_16/dropout/random_uniform/RandomUniformRandomUniform/while/simple_rnn_cell_16/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype0t
/while/simple_rnn_cell_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ё
-while/simple_rnn_cell_16/dropout/GreaterEqualGreaterEqualFwhile/simple_rnn_cell_16/dropout/random_uniform/RandomUniform:output:08while/simple_rnn_cell_16/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ё
%while/simple_rnn_cell_16/dropout/CastCast1while/simple_rnn_cell_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2Д
&while/simple_rnn_cell_16/dropout/Mul_1Mul(while/simple_rnn_cell_16/dropout/Mul:z:0)while/simple_rnn_cell_16/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2m
*while/simple_rnn_cell_16/ones_like_1/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:o
*while/simple_rnn_cell_16/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Щ
$while/simple_rnn_cell_16/ones_like_1Fill3while/simple_rnn_cell_16/ones_like_1/Shape:output:03while/simple_rnn_cell_16/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџm
(while/simple_rnn_cell_16/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Т
&while/simple_rnn_cell_16/dropout_1/MulMul-while/simple_rnn_cell_16/ones_like_1:output:01while/simple_rnn_cell_16/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
(while/simple_rnn_cell_16/dropout_1/ShapeShape-while/simple_rnn_cell_16/ones_like_1:output:0*
T0*
_output_shapes
:У
?while/simple_rnn_cell_16/dropout_1/random_uniform/RandomUniformRandomUniform1while/simple_rnn_cell_16/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0v
1while/simple_rnn_cell_16/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ј
/while/simple_rnn_cell_16/dropout_1/GreaterEqualGreaterEqualHwhile/simple_rnn_cell_16/dropout_1/random_uniform/RandomUniform:output:0:while/simple_rnn_cell_16/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџІ
'while/simple_rnn_cell_16/dropout_1/CastCast3while/simple_rnn_cell_16/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЛ
(while/simple_rnn_cell_16/dropout_1/Mul_1Mul*while/simple_rnn_cell_16/dropout_1/Mul:z:0+while/simple_rnn_cell_16/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџГ
while/simple_rnn_cell_16/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/simple_rnn_cell_16/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2Љ
.while/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	2*
dtype0Ж
while/simple_rnn_cell_16/MatMulMatMul while/simple_rnn_cell_16/mul:z:06while/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЇ
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Т
 while/simple_rnn_cell_16/BiasAddBiasAdd)while/simple_rnn_cell_16/MatMul:product:07while/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/simple_rnn_cell_16/mul_1Mulwhile_placeholder_3,while/simple_rnn_cell_16/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЎ
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0М
!while/simple_rnn_cell_16/MatMul_1MatMul"while/simple_rnn_cell_16/mul_1:z:08while/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџА
while/simple_rnn_cell_16/addAddV2)while/simple_rnn_cell_16/BiasAdd:output:0+while/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџz
while/simple_rnn_cell_16/TanhTanh while/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџe
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
while/SelectV2SelectV2while/Tile:output:0!while/simple_rnn_cell_16/Tanh:y:0while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџg
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
while/SelectV2_1SelectV2while/Tile_1:output:0!while/simple_rnn_cell_16/Tanh:y:0while_placeholder_3*
T0*(
_output_shapes
:џџџџџџџџџР
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: u
while/Identity_4Identitywhile/SelectV2:output:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
while/Identity_5Identitywhile/SelectV2_1:output:0^while/NoOp*
T0*(
_output_shapes
:џџџџџџџџџт

while/NoOpNoOp0^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_simple_rnn_cell_16_biasadd_readvariableop_resource:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_16_matmul_readvariableop_resource9while_simple_rnn_cell_16_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"А
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : 2b
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_16/MatMul/ReadVariableOp.while/simple_rnn_cell_16/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ЖQ
а
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_355828

inputs
mask
D
1simple_rnn_cell_16_matmul_readvariableop_resource:	2A
2simple_rnn_cell_16_biasadd_readvariableop_resource:	G
3simple_rnn_cell_16_matmul_1_readvariableop_resource:

identityЂ)simple_rnn_cell_16/BiasAdd/ReadVariableOpЂ(simple_rnn_cell_16/MatMul/ReadVariableOpЂ*simple_rnn_cell_16/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџm

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*+
_output_shapes
:џџџџџџџџџe
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ~
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*+
_output_shapes
:џџџџџџџџџf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maskj
"simple_rnn_cell_16/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:g
"simple_rnn_cell_16/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?А
simple_rnn_cell_16/ones_likeFill+simple_rnn_cell_16/ones_like/Shape:output:0+simple_rnn_cell_16/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2b
$simple_rnn_cell_16/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:i
$simple_rnn_cell_16/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?З
simple_rnn_cell_16/ones_like_1Fill-simple_rnn_cell_16/ones_like_1/Shape:output:0-simple_rnn_cell_16/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_16/mulMulstrided_slice_2:output:0%simple_rnn_cell_16/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
(simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_16_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype0Є
simple_rnn_cell_16/MatMulMatMulsimple_rnn_cell_16/mul:z:00simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
)simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0А
simple_rnn_cell_16/BiasAddBiasAdd#simple_rnn_cell_16/MatMul:product:01simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_16/mul_1Mulzeros:output:0'simple_rnn_cell_16/ones_like_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
*simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Њ
simple_rnn_cell_16/MatMul_1MatMulsimple_rnn_cell_16/mul_1:z:02simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_16/addAddV2#simple_rnn_cell_16/BiasAdd:output:0%simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџn
simple_rnn_cell_16/TanhTanhsimple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : h
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџИ
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ц
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШg

zeros_like	ZerosLikesimple_rnn_cell_16/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџc
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ж
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:01simple_rnn_cell_16_matmul_readvariableop_resource2simple_rnn_cell_16_biasadd_readvariableop_resource3simple_rnn_cell_16_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*P
_output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_355739*
condR
while_cond_355738*O
output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   У
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџc
IdentityIdentitytranspose_2:y:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџв
NoOpNoOp*^simple_rnn_cell_16/BiasAdd/ReadVariableOp)^simple_rnn_cell_16/MatMul/ReadVariableOp+^simple_rnn_cell_16/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџ2:џџџџџџџџџ: : : 2V
)simple_rnn_cell_16/BiasAdd/ReadVariableOp)simple_rnn_cell_16/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_16/MatMul/ReadVariableOp(simple_rnn_cell_16/MatMul/ReadVariableOp2X
*simple_rnn_cell_16/MatMul_1/ReadVariableOp*simple_rnn_cell_16/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs:MI
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask
f
б
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_358796

inputs
mask
E
1simple_rnn_cell_17_matmul_readvariableop_resource:
A
2simple_rnn_cell_17_biasadd_readvariableop_resource:	G
3simple_rnn_cell_17_matmul_1_readvariableop_resource:

identityЂ)simple_rnn_cell_17/BiasAdd/ReadVariableOpЂ(simple_rnn_cell_17/MatMul/ReadVariableOpЂ*simple_rnn_cell_17/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџm

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*+
_output_shapes
:џџџџџџџџџe
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ~
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*+
_output_shapes
:џџџџџџџџџf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskj
"simple_rnn_cell_17/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:g
"simple_rnn_cell_17/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Б
simple_rnn_cell_17/ones_likeFill+simple_rnn_cell_17/ones_like/Shape:output:0+simple_rnn_cell_17/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџe
 simple_rnn_cell_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Њ
simple_rnn_cell_17/dropout/MulMul%simple_rnn_cell_17/ones_like:output:0)simple_rnn_cell_17/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџu
 simple_rnn_cell_17/dropout/ShapeShape%simple_rnn_cell_17/ones_like:output:0*
T0*
_output_shapes
:Г
7simple_rnn_cell_17/dropout/random_uniform/RandomUniformRandomUniform)simple_rnn_cell_17/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0n
)simple_rnn_cell_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?р
'simple_rnn_cell_17/dropout/GreaterEqualGreaterEqual@simple_rnn_cell_17/dropout/random_uniform/RandomUniform:output:02simple_rnn_cell_17/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/dropout/CastCast+simple_rnn_cell_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЃ
 simple_rnn_cell_17/dropout/Mul_1Mul"simple_rnn_cell_17/dropout/Mul:z:0#simple_rnn_cell_17/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџb
$simple_rnn_cell_17/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:i
$simple_rnn_cell_17/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?З
simple_rnn_cell_17/ones_like_1Fill-simple_rnn_cell_17/ones_like_1/Shape:output:0-simple_rnn_cell_17/ones_like_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџg
"simple_rnn_cell_17/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @А
 simple_rnn_cell_17/dropout_1/MulMul'simple_rnn_cell_17/ones_like_1:output:0+simple_rnn_cell_17/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџy
"simple_rnn_cell_17/dropout_1/ShapeShape'simple_rnn_cell_17/ones_like_1:output:0*
T0*
_output_shapes
:З
9simple_rnn_cell_17/dropout_1/random_uniform/RandomUniformRandomUniform+simple_rnn_cell_17/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0p
+simple_rnn_cell_17/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ц
)simple_rnn_cell_17/dropout_1/GreaterEqualGreaterEqualBsimple_rnn_cell_17/dropout_1/random_uniform/RandomUniform:output:04simple_rnn_cell_17/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
!simple_rnn_cell_17/dropout_1/CastCast-simple_rnn_cell_17/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЉ
"simple_rnn_cell_17/dropout_1/Mul_1Mul$simple_rnn_cell_17/dropout_1/Mul:z:0%simple_rnn_cell_17/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/mulMulstrided_slice_2:output:0$simple_rnn_cell_17/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
(simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_17_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Є
simple_rnn_cell_17/MatMulMatMulsimple_rnn_cell_17/mul:z:00simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
)simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0А
simple_rnn_cell_17/BiasAddBiasAdd#simple_rnn_cell_17/MatMul:product:01simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/mul_1Mulzeros:output:0&simple_rnn_cell_17/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ 
*simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_17_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Њ
simple_rnn_cell_17/MatMul_1MatMulsimple_rnn_cell_17/mul_1:z:02simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_17/addAddV2#simple_rnn_cell_17/BiasAdd:output:0%simple_rnn_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџn
simple_rnn_cell_17/TanhTanhsimple_rnn_cell_17/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : h
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџИ
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ц
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШg

zeros_like	ZerosLikesimple_rnn_cell_17/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџc
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ж
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:01simple_rnn_cell_17_matmul_readvariableop_resource2simple_rnn_cell_17_biasadd_readvariableop_resource3simple_rnn_cell_17_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*P
_output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_358690*
condR
while_cond_358689*O
output_shapes>
<: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   з
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџв
NoOpNoOp*^simple_rnn_cell_17/BiasAdd/ReadVariableOp)^simple_rnn_cell_17/MatMul/ReadVariableOp+^simple_rnn_cell_17/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:џџџџџџџџџ:џџџџџџџџџ: : : 2V
)simple_rnn_cell_17/BiasAdd/ReadVariableOp)simple_rnn_cell_17/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_17/MatMul/ReadVariableOp(simple_rnn_cell_17/MatMul/ReadVariableOp2X
*simple_rnn_cell_17/MatMul_1/ReadVariableOp*simple_rnn_cell_17/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:MI
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask"Е	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*С
serving_default­
Q
embedding_12_input;
$serving_default_embedding_12_input:0џџџџџџџџџ<
dense_250
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ы
Љ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
Е
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
У
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec"
_tf_keras_rnn_layer
У
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$cell
%
state_spec"
_tf_keras_rnn_layer
Л
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias"
_tf_keras_layer
Л
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias"
_tf_keras_layer
n
0
61
72
83
94
:5
;6
,7
-8
49
510"
trackable_list_wrapper
n
0
61
72
83
94
:5
;6
,7
-8
49
510"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
э
Atrace_0
Btrace_1
Ctrace_2
Dtrace_32
.__inference_sequential_12_layer_call_fn_356054
.__inference_sequential_12_layer_call_fn_356736
.__inference_sequential_12_layer_call_fn_356763
.__inference_sequential_12_layer_call_fn_356608П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zAtrace_0zBtrace_1zCtrace_2zDtrace_3
й
Etrace_0
Ftrace_1
Gtrace_2
Htrace_32ю
I__inference_sequential_12_layer_call_and_return_conditional_losses_357079
I__inference_sequential_12_layer_call_and_return_conditional_losses_357459
I__inference_sequential_12_layer_call_and_return_conditional_losses_356641
I__inference_sequential_12_layer_call_and_return_conditional_losses_356674П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zEtrace_0zFtrace_1zGtrace_2zHtrace_3
зBд
!__inference__wrapped_model_355005embedding_12_input"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Џ
Iiter

Jbeta_1

Kbeta_2
	Ldecay
Mlearning_ratemЋ,mЌ-m­4mЎ5mЏ6mА7mБ8mВ9mГ:mД;mЕvЖ,vЗ-vИ4vЙ5vК6vЛ7vМ8vН9vО:vП;vР"
	optimizer
,
Nserving_default"
signature_map
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ё
Ttrace_02д
-__inference_embedding_12_layer_call_fn_357466Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zTtrace_0

Utrace_02я
H__inference_embedding_12_layer_call_and_return_conditional_losses_357476Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zUtrace_0
*:(	'22embedding_12/embeddings
5
60
71
82"
trackable_list_wrapper
5
60
71
82"
trackable_list_wrapper
 "
trackable_list_wrapper
Й

Vstates
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object

\trace_0
]trace_1
^trace_2
_trace_32
.__inference_simple_rnn_16_layer_call_fn_357487
.__inference_simple_rnn_16_layer_call_fn_357498
.__inference_simple_rnn_16_layer_call_fn_357510
.__inference_simple_rnn_16_layer_call_fn_357522д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z\trace_0z]trace_1z^trace_2z_trace_3
ю
`trace_0
atrace_1
btrace_2
ctrace_32
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_357646
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_357802
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_357951
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_358132д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z`trace_0zatrace_1zbtrace_2zctrace_3
ш
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses
j_random_generator

6kernel
7recurrent_kernel
8bias"
_tf_keras_layer
 "
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
 "
trackable_list_wrapper
Й

kstates
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object

qtrace_0
rtrace_1
strace_2
ttrace_32
.__inference_simple_rnn_17_layer_call_fn_358143
.__inference_simple_rnn_17_layer_call_fn_358154
.__inference_simple_rnn_17_layer_call_fn_358166
.__inference_simple_rnn_17_layer_call_fn_358178д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zqtrace_0zrtrace_1zstrace_2zttrace_3
ю
utrace_0
vtrace_1
wtrace_2
xtrace_32
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_358304
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_358462
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_358613
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_358796д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zutrace_0zvtrace_1zwtrace_2zxtrace_3
ш
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
_random_generator

9kernel
:recurrent_kernel
;bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
я
trace_02а
)__inference_dense_24_layer_call_fn_358805Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ы
D__inference_dense_24_layer_call_and_return_conditional_losses_358816Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
": 	@2dense_24/kernel
:@2dense_24/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
я
trace_02а
)__inference_dense_25_layer_call_fn_358825Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ы
D__inference_dense_25_layer_call_and_return_conditional_losses_358836Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
!:@2dense_25/kernel
:2dense_25/bias
::8	22'simple_rnn_16/simple_rnn_cell_16/kernel
E:C
21simple_rnn_16/simple_rnn_cell_16/recurrent_kernel
4:22%simple_rnn_16/simple_rnn_cell_16/bias
;:9
2'simple_rnn_17/simple_rnn_cell_17/kernel
E:C
21simple_rnn_17/simple_rnn_cell_17/recurrent_kernel
4:22%simple_rnn_17/simple_rnn_cell_17/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_12_layer_call_fn_356054embedding_12_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
.__inference_sequential_12_layer_call_fn_356736inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
.__inference_sequential_12_layer_call_fn_356763inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
.__inference_sequential_12_layer_call_fn_356608embedding_12_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
I__inference_sequential_12_layer_call_and_return_conditional_losses_357079inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
I__inference_sequential_12_layer_call_and_return_conditional_losses_357459inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ІBЃ
I__inference_sequential_12_layer_call_and_return_conditional_losses_356641embedding_12_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ІBЃ
I__inference_sequential_12_layer_call_and_return_conditional_losses_356674embedding_12_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
жBг
$__inference_signature_wrapper_356709embedding_12_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBо
-__inference_embedding_12_layer_call_fn_357466inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
H__inference_embedding_12_layer_call_and_return_conditional_losses_357476inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_simple_rnn_16_layer_call_fn_357487inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
.__inference_simple_rnn_16_layer_call_fn_357498inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
.__inference_simple_rnn_16_layer_call_fn_357510inputsmask"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
.__inference_simple_rnn_16_layer_call_fn_357522inputsmask"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
БBЎ
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_357646inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
БBЎ
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_357802inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЕBВ
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_357951inputsmask"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЕBВ
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_358132inputsmask"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
5
60
71
82"
trackable_list_wrapper
5
60
71
82"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
х
trace_0
trace_12Њ
3__inference_simple_rnn_cell_16_layer_call_fn_358850
3__inference_simple_rnn_cell_16_layer_call_fn_358864Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12р
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_358889
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_358930Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
$0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_simple_rnn_17_layer_call_fn_358143inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
.__inference_simple_rnn_17_layer_call_fn_358154inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
.__inference_simple_rnn_17_layer_call_fn_358166inputsmask"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
.__inference_simple_rnn_17_layer_call_fn_358178inputsmask"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
БBЎ
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_358304inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
БBЎ
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_358462inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЕBВ
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_358613inputsmask"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЕBВ
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_358796inputsmask"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
5
90
:1
;2"
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
х
trace_0
trace_12Њ
3__inference_simple_rnn_cell_17_layer_call_fn_358944
3__inference_simple_rnn_cell_17_layer_call_fn_358958Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

 trace_0
Ёtrace_12р
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_358983
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_359024Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z trace_0zЁtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
нBк
)__inference_dense_24_layer_call_fn_358805inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
D__inference_dense_24_layer_call_and_return_conditional_losses_358816inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
нBк
)__inference_dense_25_layer_call_fn_358825inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
D__inference_dense_25_layer_call_and_return_conditional_losses_358836inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
Ђ	variables
Ѓ	keras_api

Єtotal

Ѕcount"
_tf_keras_metric
c
І	variables
Ї	keras_api

Јtotal

Љcount
Њ
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
3__inference_simple_rnn_cell_16_layer_call_fn_358850inputsstates/0"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
3__inference_simple_rnn_cell_16_layer_call_fn_358864inputsstates/0"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЇBЄ
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_358889inputsstates/0"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЇBЄ
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_358930inputsstates/0"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
3__inference_simple_rnn_cell_17_layer_call_fn_358944inputsstates/0"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
3__inference_simple_rnn_cell_17_layer_call_fn_358958inputsstates/0"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЇBЄ
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_358983inputsstates/0"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЇBЄ
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_359024inputsstates/0"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
Є0
Ѕ1"
trackable_list_wrapper
.
Ђ	variables"
_generic_user_object
:  (2total
:  (2count
0
Ј0
Љ1"
trackable_list_wrapper
.
І	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
/:-	'22Adam/embedding_12/embeddings/m
':%	@2Adam/dense_24/kernel/m
 :@2Adam/dense_24/bias/m
&:$@2Adam/dense_25/kernel/m
 :2Adam/dense_25/bias/m
?:=	22.Adam/simple_rnn_16/simple_rnn_cell_16/kernel/m
J:H
28Adam/simple_rnn_16/simple_rnn_cell_16/recurrent_kernel/m
9:72,Adam/simple_rnn_16/simple_rnn_cell_16/bias/m
@:>
2.Adam/simple_rnn_17/simple_rnn_cell_17/kernel/m
J:H
28Adam/simple_rnn_17/simple_rnn_cell_17/recurrent_kernel/m
9:72,Adam/simple_rnn_17/simple_rnn_cell_17/bias/m
/:-	'22Adam/embedding_12/embeddings/v
':%	@2Adam/dense_24/kernel/v
 :@2Adam/dense_24/bias/v
&:$@2Adam/dense_25/kernel/v
 :2Adam/dense_25/bias/v
?:=	22.Adam/simple_rnn_16/simple_rnn_cell_16/kernel/v
J:H
28Adam/simple_rnn_16/simple_rnn_cell_16/recurrent_kernel/v
9:72,Adam/simple_rnn_16/simple_rnn_cell_16/bias/v
@:>
2.Adam/simple_rnn_17/simple_rnn_cell_17/kernel/v
J:H
28Adam/simple_rnn_17/simple_rnn_cell_17/recurrent_kernel/v
9:72,Adam/simple_rnn_17/simple_rnn_cell_17/bias/vЄ
!__inference__wrapped_model_3550056879;:,-45;Ђ8
1Ђ.
,)
embedding_12_inputџџџџџџџџџ
Њ "3Њ0
.
dense_25"
dense_25џџџџџџџџџЅ
D__inference_dense_24_layer_call_and_return_conditional_losses_358816],-0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ@
 }
)__inference_dense_24_layer_call_fn_358805P,-0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ@Є
D__inference_dense_25_layer_call_and_return_conditional_losses_358836\45/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ
 |
)__inference_dense_25_layer_call_fn_358825O45/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџЋ
H__inference_embedding_12_layer_call_and_return_conditional_losses_357476_/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ2
 
-__inference_embedding_12_layer_call_fn_357466R/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ2Ц
I__inference_sequential_12_layer_call_and_return_conditional_losses_356641y6879;:,-45CЂ@
9Ђ6
,)
embedding_12_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Ц
I__inference_sequential_12_layer_call_and_return_conditional_losses_356674y6879;:,-45CЂ@
9Ђ6
,)
embedding_12_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 К
I__inference_sequential_12_layer_call_and_return_conditional_losses_357079m6879;:,-457Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 К
I__inference_sequential_12_layer_call_and_return_conditional_losses_357459m6879;:,-457Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
.__inference_sequential_12_layer_call_fn_356054l6879;:,-45CЂ@
9Ђ6
,)
embedding_12_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
.__inference_sequential_12_layer_call_fn_356608l6879;:,-45CЂ@
9Ђ6
,)
embedding_12_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
.__inference_sequential_12_layer_call_fn_356736`6879;:,-457Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
.__inference_sequential_12_layer_call_fn_356763`6879;:,-457Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџО
$__inference_signature_wrapper_3567096879;:,-45QЂN
Ђ 
GЊD
B
embedding_12_input,)
embedding_12_inputџџџџџџџџџ"3Њ0
.
dense_25"
dense_25џџџџџџџџџй
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_357646687OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ2

 
p 

 
Њ "3Ђ0
)&
0џџџџџџџџџџџџџџџџџџ
 й
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_357802687OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ2

 
p

 
Њ "3Ђ0
)&
0џџџџџџџџџџџџџџџџџџ
 м
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_357951687[ЂX
QЂN
$!
inputsџџџџџџџџџ2

maskџџџџџџџџџ

p 

 
Њ "*Ђ'
 
0џџџџџџџџџ
 м
I__inference_simple_rnn_16_layer_call_and_return_conditional_losses_358132687[ЂX
QЂN
$!
inputsџџџџџџџџџ2

maskџџџџџџџџџ

p

 
Њ "*Ђ'
 
0џџџџџџџџџ
 А
.__inference_simple_rnn_16_layer_call_fn_357487~687OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ2

 
p 

 
Њ "&#џџџџџџџџџџџџџџџџџџА
.__inference_simple_rnn_16_layer_call_fn_357498~687OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ2

 
p

 
Њ "&#џџџџџџџџџџџџџџџџџџД
.__inference_simple_rnn_16_layer_call_fn_357510687[ЂX
QЂN
$!
inputsџџџџџџџџџ2

maskџџџџџџџџџ

p 

 
Њ "џџџџџџџџџД
.__inference_simple_rnn_16_layer_call_fn_357522687[ЂX
QЂN
$!
inputsџџџџџџџџџ2

maskџџџџџџџџџ

p

 
Њ "џџџџџџџџџЬ
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_3583049;:PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "&Ђ#

0џџџџџџџџџ
 Ь
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_3584629;:PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "&Ђ#

0џџџџџџџџџ
 й
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_3586139;:\ЂY
RЂO
%"
inputsџџџџџџџџџ

maskџџџџџџџџџ

p 

 
Њ "&Ђ#

0џџџџџџџџџ
 й
I__inference_simple_rnn_17_layer_call_and_return_conditional_losses_3587969;:\ЂY
RЂO
%"
inputsџџџџџџџџџ

maskџџџџџџџџџ

p

 
Њ "&Ђ#

0џџџџџџџџџ
 Є
.__inference_simple_rnn_17_layer_call_fn_358143r9;:PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџЄ
.__inference_simple_rnn_17_layer_call_fn_358154r9;:PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџА
.__inference_simple_rnn_17_layer_call_fn_358166~9;:\ЂY
RЂO
%"
inputsџџџџџџџџџ

maskџџџџџџџџџ

p 

 
Њ "џџџџџџџџџА
.__inference_simple_rnn_17_layer_call_fn_358178~9;:\ЂY
RЂO
%"
inputsџџџџџџџџџ

maskџџџџџџџџџ

p

 
Њ "џџџџџџџџџ
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_358889К687]ЂZ
SЂP
 
inputsџџџџџџџџџ2
(Ђ%
# 
states/0џџџџџџџџџ
p 
Њ "TЂQ
JЂG

0/0џџџџџџџџџ
%"
 
0/1/0џџџџџџџџџ
 
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_358930К687]ЂZ
SЂP
 
inputsџџџџџџџџџ2
(Ђ%
# 
states/0џџџџџџџџџ
p
Њ "TЂQ
JЂG

0/0џџџџџџџџџ
%"
 
0/1/0џџџџџџџџџ
 ф
3__inference_simple_rnn_cell_16_layer_call_fn_358850Ќ687]ЂZ
SЂP
 
inputsџџџџџџџџџ2
(Ђ%
# 
states/0џџџџџџџџџ
p 
Њ "FЂC

0џџџџџџџџџ
# 

1/0џџџџџџџџџф
3__inference_simple_rnn_cell_16_layer_call_fn_358864Ќ687]ЂZ
SЂP
 
inputsџџџџџџџџџ2
(Ђ%
# 
states/0џџџџџџџџџ
p
Њ "FЂC

0џџџџџџџџџ
# 

1/0џџџџџџџџџ
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_358983Л9;:^Ђ[
TЂQ
!
inputsџџџџџџџџџ
(Ђ%
# 
states/0џџџџџџџџџ
p 
Њ "TЂQ
JЂG

0/0џџџџџџџџџ
%"
 
0/1/0џџџџџџџџџ
 
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_359024Л9;:^Ђ[
TЂQ
!
inputsџџџџџџџџџ
(Ђ%
# 
states/0џџџџџџџџџ
p
Њ "TЂQ
JЂG

0/0џџџџџџџџџ
%"
 
0/1/0џџџџџџџџџ
 х
3__inference_simple_rnn_cell_17_layer_call_fn_358944­9;:^Ђ[
TЂQ
!
inputsџџџџџџџџџ
(Ђ%
# 
states/0џџџџџџџџџ
p 
Њ "FЂC

0џџџџџџџџџ
# 

1/0џџџџџџџџџх
3__inference_simple_rnn_cell_17_layer_call_fn_358958­9;:^Ђ[
TЂQ
!
inputsџџџџџџџџџ
(Ђ%
# 
states/0џџџџџџџџџ
p
Њ "FЂC

0џџџџџџџџџ
# 

1/0џџџџџџџџџ