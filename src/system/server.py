import socket
import struct
import json
import numpy as np
import tensorflow as tf
from utils import DataFactory

#socket.setdefaulttimeout(20)
def read_bytes(sock, size):
    buf = b""
    while len(buf) != size:
        ret = sock.recv(size - len(buf))
        if not ret:
            raise Exception("Socket closed")
        buf += ret
    return buf


def read_num(sock):
    size = struct.calcsize("=L")
    data = read_bytes(sock, size)
    num = struct.unpack("=L", data)
    num = int(num[0])
    return num
def deal_input(js):
    num_frames = len(js)
    shape = (num_frames, 13)
    data = np.zeros((shape))
    for frame in range(num_frames):
        index = 'f' + str(frame)
        get = js[index]
        get[:] = [float(each) for each in get]
        data[frame, :] = get
    return data 

def deal_output(out):
    js = dict()
    num_frames = out.shape[0]
    for frame in range(num_frames):
        index = 'f' + str(frame)
        get = list(out[frame])
        get[:] = [str(each) for each in get]
        js[index] = get
    # print("JS", js)
    return js



def generate_defensive_strategy(sess, graph, offense_input):
    """ Given one offensive input, generate 100 defensive strategies, and reture only one result with the hightest score . 

    Inputs 
    ------
    offense_input : float, shape=[lenght,13]
        lenght could be variable, [13] -> [ball's xyz * 1, offensive player's xy * 5]

    Returns
    -------
    defense_result : float, shape=[length,10]
        lenght could be variable, [10] -> [defensive player's xy * 5]
    """

    # placeholder tensor
    latent_input_t = graph.get_tensor_by_name('Generator/latent_input:0')
    team_a_t = graph.get_tensor_by_name('Generator/team_a:0')
    G_samples_t = graph.get_tensor_by_name('Critic/G_samples:0')
    matched_cond_t = graph.get_tensor_by_name('Critic/matched_cond:0')
    # result tensor
    result_t = graph.get_tensor_by_name(
        'Generator/G_inference/conv_result/conv1d/Maximum:0')
    critic_scores_t = graph.get_tensor_by_name(
        'Critic/C_inference_1/conv_output/Reshape:0')

    real_data = np.load('../../data/FEATURES-4.npy')
    # DataFactory
    data_factory = DataFactory(real_data)
    conditions = data_factory.normalize_offense(
        np.expand_dims(offense_input, axis=0))
    # given 100 latents generate 100 results on same condition at once
    conditions_duplicated = np.concatenate(
        [conditions for _ in range(100)], axis=0)
    # generate result
    latents = np.random.normal(
        0., 1., size=[100, 100])
    feed_dict = {
        latent_input_t: latents,
        team_a_t: conditions_duplicated
    }
    result = sess.run(
        result_t, feed_dict=feed_dict)
    # calculate em distance
    feed_dict = {
        G_samples_t: result,
        matched_cond_t: conditions_duplicated
    }
    critic_scores = sess.run(
        critic_scores_t, feed_dict=feed_dict)
    recoverd_A_fake_B = data_factory.recover_B(result)

    return recoverd_A_fake_B[np.argmax(critic_scores)]


def main():

    ##### restore model #####

    with tf.get_default_graph().as_default() as graph:
        # sesstion config
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.import_meta_graph('ckpt/model.ckpt-123228.meta')
        with tf.Session(config=config) as sess:
            # restored
            saver.restore(sess, 'ckpt/model.ckpt-123228')
            print('Congrat! Successfully restore model :D')

            # offense_input = np.zeros(shape=[123, 13])  # an example
            # defense_result = generate_defensive_strategy(
            #     sess, graph, offense_input)

            # print(defense_result.shape)
            # exit()

            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            host = socket.gethostbyname('140.113.210.4')
            port = 5000

            server_socket.bind((host, port))
            server_socket.listen(5)

            while True:
                try:
                    client_socket, addr = server_socket.accept()
                    print("Connection from {}".format(str(addr)))

                    # Receive input json
                    datasize = read_num(client_socket)
                    print(type(datasize), datasize)
                    data = read_bytes(client_socket, datasize)
                    jdata = json.loads(data.decode())

                    ####Do something#####
                    # offense_input = np.zeros(shape=[123, 13])  # an example
                    # defense_result = generate_defensive_strategy(
                    #     sess, graph, offense_input)
                    # print(defense_result.shape)
                    #####################

                    offense_input = deal_input(jdata)
                    defense_result = generate_defensive_strategy(
                        sess, graph, offense_input
                    )
                    output_send = deal_output(defense_result)
                    #output_send = list(output_send)
                    # print("type", type(output_send))
                    # print("output", output_send)
                    # send output json
                    output = json.dumps(output_send)
                    client_socket.sendall(struct.pack("=L", len(output.encode())))
                    client_socket.sendall(output.encode())
                    print("Send!")
                # client_socket.close()
                except:
                    print("Failure Connection!")


if __name__ == '__main__':
    main()
