"""
BezierGAN for capturing the airfoil manifold

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
import tensorflow as tf

from shape_plot import plot_grid


def preprocess(X):
    X = np.expand_dims(X, axis=-1)
    return X.astype(np.float32)

def postprocess(X):
    X = np.squeeze(X)
    return X

EPSILON = 1e-7

class GAN(object):
    
    def __init__(self, latent_dim=2, noise_dim=100, n_points=64, bezier_degree=16, bounds=(0.0, 1.0)):

        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        
        self.X_shape = (n_points, 2, 1)
        self.bezier_degree = bezier_degree
        self.bounds = bounds
        
    def generator(self, c, z, reuse=tf.AUTO_REUSE, training=True):
        
        depth_cpw = 32*8
        dim_cpw = (self.bezier_degree+1)/8
        kernel_size = (4,3)
#        noise_std = 0.01
        
        with tf.variable_scope('Generator', reuse=reuse):
                
            if self.noise_dim == 0:
                cz = c
            else:
                cz = tf.concat([c, z], axis=-1)
            
            cpw = tf.layers.dense(cz, 1024)
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
    
            cpw = tf.layers.dense(cpw, dim_cpw*3*depth_cpw)
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
            cpw = tf.reshape(cpw, (-1, dim_cpw, 3, depth_cpw))
    
            cpw = tf.layers.conv2d_transpose(cpw, int(depth_cpw/2), kernel_size, strides=(2,1), padding='same')
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
#            cpw += tf.random_normal(shape=tf.shape(cpw), stddev=noise_std)
            
            cpw = tf.layers.conv2d_transpose(cpw, int(depth_cpw/4), kernel_size, strides=(2,1), padding='same')
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
#            cpw += tf.random_normal(shape=tf.shape(cpw), stddev=noise_std)
            
            cpw = tf.layers.conv2d_transpose(cpw, int(depth_cpw/8), kernel_size, strides=(2,1), padding='same')
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
#            cpw += tf.random_normal(shape=tf.shape(cpw), stddev=noise_std)
            
            # Control points
            cp = tf.layers.conv2d(cpw, 1, (1,2), padding='valid') # batch_size x (bezier_degree+1) x 2 x 1
            cp = tf.nn.tanh(cp)
            cp = tf.squeeze(cp, axis=-1, name='control_point') # batch_size x (bezier_degree+1) x 2
            
            # Weights
            w = tf.layers.conv2d(cpw, 1, (1,3), padding='valid')
            w = tf.nn.sigmoid(w) # batch_size x (bezier_degree+1) x 1 x 1
            w = tf.squeeze(w, axis=-1, name='weight') # batch_size x (bezier_degree+1) x 1
            
            # Parameters at data points
            db = tf.layers.dense(cz, 1024)
            db = tf.layers.batch_normalization(db, momentum=0.9)
            db = tf.nn.leaky_relu(db, alpha=0.2)
            
            db = tf.layers.dense(db, 256)
            db = tf.layers.batch_normalization(db, momentum=0.9)
            db = tf.nn.leaky_relu(db, alpha=0.2)
            
            db = tf.layers.dense(db, self.X_shape[0]-1)
            db = tf.nn.softmax(db) # batch_size x (n_data_points-1)
            
#            db = tf.random_gamma([tf.shape(cz)[0], self.X_shape[0]-1], alpha=100, beta=100)
#            db = tf.nn.softmax(db) # batch_size x (n_data_points-1)
            
            ub = tf.pad(db, [[0,0],[1,0]], constant_values=0) # batch_size x n_data_points
            ub = tf.cumsum(ub, axis=1)
            ub = tf.minimum(ub, 1)
            ub = tf.expand_dims(ub, axis=-1) # 1 x n_data_points x 1
            
            # Bezier layer
            # Compute values of basis functions at data points
            num_control_points = self.bezier_degree + 1
            lbs = tf.tile(ub, [1, 1, num_control_points]) # batch_size x n_data_points x n_control_points
            pw1 = tf.range(0, num_control_points, dtype=tf.float32)
            pw1 = tf.reshape(pw1, [1, 1, -1]) # 1 x 1 x n_control_points
            pw2 = tf.reverse(pw1, axis=[-1])
            lbs = tf.add(tf.multiply(pw1, tf.log(lbs+EPSILON)), tf.multiply(pw2, tf.log(1-lbs+EPSILON))) # batch_size x n_data_points x n_control_points
            lc = tf.add(tf.lgamma(pw1+1), tf.lgamma(pw2+1))
            lc = tf.subtract(tf.lgamma(tf.cast(num_control_points, dtype=tf.float32)), lc) # 1 x 1 x n_control_points
            lbs = tf.add(lbs, lc) # batch_size x n_data_points x n_control_points
            bs = tf.exp(lbs)
            # Compute data points
            cp_w = tf.multiply(cp, w)
            dp = tf.matmul(bs, cp_w) # batch_size x n_data_points x 2
            bs_w = tf.matmul(bs, w) # batch_size x n_data_points x 1
            dp = tf.div(dp, bs_w) # batch_size x n_data_points x 2
            dp = tf.expand_dims(dp, axis=-1, name='fake_image') # batch_size x n_data_points x 2 x 1
            
            return dp, cp, w, ub, db
        
    def discriminator(self, x, reuse=tf.AUTO_REUSE, training=True):
        
        depth = 64
        dropout = 0.4
        kernel_size = (4,2)
        
        with tf.variable_scope('Discriminator', reuse=reuse):
        
            x = tf.layers.conv2d(x, depth*1, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*2, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*4, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*8, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*16, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*32, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 1024)
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            d = tf.layers.dense(x, 1)
            
            q = tf.layers.dense(x, 128)
            q = tf.layers.batch_normalization(q, momentum=0.9)
            q = tf.nn.leaky_relu(q, alpha=0.2)
            q_mean = tf.layers.dense(q, self.latent_dim)
            q_logstd = tf.layers.dense(q, self.latent_dim)
            q_logstd = tf.maximum(q_logstd, -16)
            # Reshape to batch_size x 1 x latent_dim
            q_mean = tf.reshape(q_mean, (-1, 1, self.latent_dim))
            q_logstd = tf.reshape(q_logstd, (-1, 1, self.latent_dim))
            q = tf.concat([q_mean, q_logstd], axis=1, name='predicted_latent') # batch_size x 2 x latent_dim
            
            return d, q
        
    def train(self, X_train, train_steps=2000, batch_size=256, save_interval=0):
            
        X_train = preprocess(X_train)
        
        # Inputs
        self.x = tf.placeholder(tf.float32, shape=(None,)+self.X_shape, name='real_image')
        self.c = tf.placeholder(tf.float32, shape=[None, self.latent_dim], name='latent_code')
        self.z = tf.placeholder(tf.float32, shape=[None, self.noise_dim], name='noise')
        
        # Targets
        q_target = tf.placeholder(tf.float32, shape=[None, self.latent_dim])
        
        # Outputs
        d_real, _ = self.discriminator(self.x)
        x_fake_train, cp_train, w_train, ub_train, db_train = self.generator(self.c, self.z)
        d_fake, q_fake_train = self.discriminator(x_fake_train)
        
        self.x_fake_test, self.cp, self.w, ub, db = self.generator(self.c, self.z, training=False)
        _, self.q_test = self.discriminator(self.x, training=False)
        
        # Losses
        # Cross entropy losses for D
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
        # Cross entropy losses for G
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
        # Regularization for w, cp, a, and b
        r_w_loss = tf.reduce_mean(w_train[:,1:-1], axis=[1,2])
        cp_dist = tf.norm(cp_train[:,1:]-cp_train[:,:-1], axis=-1)
        r_cp_loss = tf.reduce_mean(cp_dist, axis=-1)
        r_cp_loss1 = tf.reduce_max(cp_dist, axis=-1)
        ends = cp_train[:,0] - cp_train[:,-1]
        r_ends_loss = tf.norm(ends, axis=-1) + tf.maximum(0.0, -10*ends[:,1])
        r_db_loss = tf.reduce_mean(db_train*tf.log(db_train), axis=-1)
        r_loss = r_w_loss + r_cp_loss + 0*r_cp_loss1 + r_ends_loss + 0*r_db_loss
        r_loss = tf.reduce_mean(r_loss)
        # Gaussian loss for Q
        q_mean = q_fake_train[:, 0, :]
        q_logstd = q_fake_train[:, 1, :]
        epsilon = (q_target - q_mean) / (tf.exp(q_logstd) + EPSILON)
        q_loss = q_logstd + 0.5 * tf.square(epsilon)
        q_loss = tf.reduce_mean(q_loss)
        
        # Optimizers
        d_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5)
        
        # Generator variables
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        # Discriminator variables
        dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        
        # Training operations
        d_train_real = d_optimizer.minimize(d_loss_real, var_list=dis_vars)
        d_train_fake = d_optimizer.minimize(d_loss_fake + q_loss, var_list=dis_vars)
        g_train = g_optimizer.minimize(g_loss + r_loss + q_loss, var_list=gen_vars)
        
#        def clip_gradient(optimizer, loss, var_list):
#            grads_and_vars = optimizer.compute_gradients(loss, var_list)
#            clipped_grads_and_vars = [(grad, var) if grad is None else 
#                                      (tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
#            train_op = optimizer.apply_gradients(clipped_grads_and_vars)
#            return train_op
#        
#        d_train_real = clip_gradient(d_optimizer, d_loss_real, dis_vars)
#        d_train_fake = clip_gradient(d_optimizer, d_loss_fake + q_loss, dis_vars)
#        g_train = clip_gradient(g_optimizer, g_loss + q_loss, gen_vars)
        
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        
        # Create summaries to monitor losses
        tf.summary.scalar('D_loss_for_real', d_loss_real)
        tf.summary.scalar('D_loss_for_fake', d_loss_fake)
        tf.summary.scalar('G_loss', g_loss)
        tf.summary.scalar('R_loss', r_loss)
        tf.summary.scalar('Q_loss', q_loss)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Start training
        self.sess = tf.Session()
        
        # Run the initializer
        self.sess.run(init)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('trained_gan/logs', graph=self.sess.graph)
    
        for t in range(train_steps):
            
            sigma = 0#0.1*(1 - t/(train_steps-1)) # linearly annealed instance noise
    
            ind = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
            X_real = X_train[ind]
            if sigma > 0:
                X_real_noise = X_real + np.random.normal(scale=sigma, size=X_real.shape)
                _, dlr = self.sess.run([d_train_real, d_loss_real],
                                       feed_dict={self.x: X_real_noise})
            else:
                _, dlr = self.sess.run([d_train_real, d_loss_real],
                                       feed_dict={self.x: X_real})
            y_latent = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(batch_size, self.latent_dim))
            noise = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim))
            X_fake = self.sess.run(self.x_fake_test, feed_dict={self.c: y_latent, self.z: noise})
            
            if np.any(np.isnan(X_fake)):
                ind = np.any(np.isnan(X_fake), axis=(1,2,3))
                print(self.sess.run(ub, feed_dict={self.c: y_latent, self.z: noise})[ind])
                assert not np.any(np.isnan(X_fake))
                
            if sigma > 0:
                X_fake_noise = X_fake + np.random.normal(scale=sigma, size=X_fake.shape)
                _, dlf, qdl = self.sess.run([d_train_fake, d_loss_fake, q_loss],
                                            feed_dict={x_fake_train: X_fake_noise, q_target: y_latent})
            else:
                _, dlf, qdl = self.sess.run([d_train_fake, d_loss_fake, q_loss],
                                            feed_dict={x_fake_train: X_fake, q_target: y_latent})
                
            y_latent = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(batch_size, self.latent_dim))
            noise = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim))
            
            _, gl, rl, qgl = self.sess.run([g_train, g_loss, r_loss, q_loss],
                                           feed_dict={self.c: y_latent, self.z: noise, q_target: y_latent})
            
            summary_str = self.sess.run(merged_summary_op, feed_dict={self.x: X_real, x_fake_train: X_fake,
                                                                      self.c: y_latent, self.z: noise, q_target: y_latent})
            
            summary_writer.add_summary(summary_str, t+1)
            
            # Show messages
            log_mesg = "%d: [D] real %f fake %f q %f" % (t+1, dlr, dlf, qdl)
            log_mesg = "%s  [G] fake %f reg %f q %f" % (log_mesg, gl, rl, qgl)
            print(log_mesg)
            
            if save_interval>0 and (t+1)%save_interval==0:
                
                from matplotlib import pyplot as plt
                
                ub_batch, db_batch = self.sess.run([ub, db], feed_dict={self.c: y_latent, self.z: noise})
                
                xx = np.linspace(0, 1, self.X_shape[0])
                plt.figure()
                for u in np.squeeze(ub_batch):
                    plt.plot(xx, u)
                plt.savefig('gan/ub.svg')
                
                plt.figure()
                for d in db_batch:
                    plt.plot(xx[:-1], d)
                plt.savefig('gan/db.svg')
                
                # Save the variables to disk.
                save_path = saver.save(self.sess, 'trained_gan/model')
                print('Model saved in path: %s' % save_path)
                print('Plotting results ...')
                plot_grid(5, gen_func=self.synthesize, d=self.latent_dim, bounds=self.bounds,
                          scale=.95, scatter=True, s=1, alpha=.7, fname='gan/synthesized')
                    
    def restore(self):
        
        self.sess = tf.Session()
        # Load meta graph and restore weights
        saver = tf.train.import_meta_graph('trained_gan/model.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint('trained_gan/'))
        
        # Access and create placeholders variables            
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name('real_image:0')
        self.c = graph.get_tensor_by_name('latent_code:0')
        self.z = graph.get_tensor_by_name('noise:0')
        self.x_fake_test = graph.get_tensor_by_name('Generator_1/fake_image:0')
        self.cp = graph.get_tensor_by_name('Generator_1/control_point:0')
        self.w = graph.get_tensor_by_name('Generator_1/weight:0')
        self.q_test = graph.get_tensor_by_name('Discriminator_2/predicted_latent:0')

    def synthesize(self, latent, noise=None):
        if isinstance(latent, int):
            N = latent
            latent = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(N, self.latent_dim))
            noise = np.random.normal(scale=0.5, size=(N, self.noise_dim))
            X, P, W = self.sess.run([self.x_fake_test, self.cp, self.w], feed_dict={self.c: latent, self.z: noise})
        else:
            N = latent.shape[0]
            if noise is None:
                noise = np.zeros((N, self.noise_dim))
            X, P, W = self.sess.run([self.x_fake_test, self.cp, self.w], feed_dict={self.c: latent, self.z: noise})
        return postprocess(X)
    
    def embed(self, X):
        latent = self.sess.run(self.q_test, feed_dict={self.x: X})
        return latent[:,0,:]
    
    