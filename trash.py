import tensorflow as tf


tm = tf.keras.models.load_model('./saved_target_model')
tm.summary()


m  = tf.keras.models.load_model('./saved_model')
m.summary()
