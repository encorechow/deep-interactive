import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def data_to_record(data, label, writer):
    h, w, c = data.shape
    data_raw = data.tostring()
    label_raw = label.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(h),
        'width': _int64_feature(w),
        'depth': _int64_feature(c),
        'label': _bytes_feature(label_raw),
        'data': _bytes_feature(data_raw)
        }))

    writer.write(example.SerializeToString())



