import os
from datetime import datetime
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# ---- Logging for TensorBoard (Lecture page 16) ----
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join("logs", "task1_1", stamp)
writer = tf.summary.create_file_writer(logdir)

# ---- constants (Lecture pages 17-18) ----
a = tf.constant(3.0, name="a")
b = tf.constant(2.0, name="b")

# ---- variables (Lecture pages 20-24) ----
w = tf.Variable(1.5, dtype=tf.float32, name="w")
c = tf.Variable(-0.5, dtype=tf.float32, name="c")

# ---- Build TF2 graph using @tf.function (Lecture page 9, 16) ----
tf.summary.trace_on(graph=True, profiler=False)

@tf.function
def build_graph(x):
    # > 10 operations total
    op1 = tf.add(x, a, name="add1")
    op2 = tf.multiply(op1, w, name="mul1")
    op3 = tf.subtract(op2, b, name="sub1")
    op4 = tf.pow(op3, 2.0, name="pow1")
    op5 = tf.add(op4, c, name="add2")
    op6 = tf.math.sqrt(tf.abs(op5), name="sqrt1")
    op7 = tf.divide(op6, b, name="div1")
    op8 = tf.math.sin(op7, name="sin1")
    op9 = tf.math.cos(op7, name="cos1")
    op10 = tf.add(op8, op9, name="add3")
    op11 = tf.reduce_sum(op10, name="sum1")
    return op11

# run computation and print
x_input = tf.constant([1.0, 2.0, 3.0], name="x_input")
result = build_graph(x_input)
print("Result:", result.numpy())

# export graph to TensorBoard
with writer.as_default():
    tf.summary.trace_export(
        name="task1_1_trace",
        step=0,
        profiler_outdir=logdir
    )

writer.close()
print("TensorBoard logdir:", logdir)
