import tensorflow as tf

w = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)

#inputs and outputs
x = tf.placeholder(tf.float32)

linear_model = w * x + b

y = tf.placeholder(tf.float32)

#loss
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)

#Optimizer
Optimizer = tf.train.GradientDescentOptimizer(0.01)
train = Optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
	sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

#print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

print(sess.run([w,b]))
