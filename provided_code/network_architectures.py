# ===== THƯ VIỆN =====
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, BatchNormalization, LeakyReLU, Dropout, concatenate, Activation
from tensorflow.keras.optimizers import Optimizer
from provided_code.data_shapes import DataShapes

# ===== KHỐI TẠO BLOCK RESIDUAL =====
def residual_block(x, filters):
    shortcut = x
    x = Conv3D(filters, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv3D(filters, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv3D(filters, kernel_size=1, padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = layers.add([x, shortcut])
    x = LeakyReLU()(x)
    return x

# ===== ATTENTION GATE =====
def attention_gate(x, g, inter_channels):
    theta_x = Conv3D(inter_channels, kernel_size=1, padding="same")(x)
    phi_g = Conv3D(inter_channels, kernel_size=1, padding="same")(g)
    add = layers.add([theta_x, phi_g])
    act = LeakyReLU()(add)
    psi = Conv3D(1, kernel_size=1, padding="same")(act)
    psi = Activation('sigmoid')(psi)
    return layers.multiply([x, psi])

# ===== MÔ HÌNH ATTENTION RESIDUAL U-NET 3D =====
class DefineDoseFromCT:
    def __init__(self, data_shapes: DataShapes, initial_number_of_filters: int, filter_size: tuple[int, int, int], stride_size: tuple[int, int, int], gen_optimizer: Optimizer):
        self.data_shapes = data_shapes
        self.initial_number_of_filters = initial_number_of_filters
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.gen_optimizer = gen_optimizer

    def define_generator(self) -> Model:
        # Đầu vào gồm CT và mask ROI
        ct_input = Input(shape=self.data_shapes.ct, name="ct")
        roi_input = Input(shape=self.data_shapes.structure_masks, name="roi_masks")
        x_input = concatenate([ct_input, roi_input])

        nf = self.initial_number_of_filters

        # Encoder (mã hóa)
        e1 = residual_block(x_input, nf)
        x = Conv3D(nf*2, 3, strides=2, padding="same")(e1)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)

        e2 = residual_block(x, nf*2)
        x = Conv3D(nf*4, 3, strides=2, padding="same")(e2)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)

        e3 = residual_block(x, nf*4)
        x = Conv3D(nf*8, 3, strides=2, padding="same")(e3)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)

        # Bottleneck (trung tâm)
        b = residual_block(x, nf*8)

        # Kết nối tắt từ input xuống bottleneck
        down_input = Conv3D(nf, kernel_size=3, strides=4, padding="same")(x_input)
        b = concatenate([b, down_input])

        # Decoder (giải mã)
        x = Conv3DTranspose(nf*4, 3, strides=2, padding="same")(b)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        x = concatenate([x, attention_gate(e3, x, nf*2)])
        x = residual_block(x, nf*4)

        x = Conv3DTranspose(nf*2, 3, strides=2, padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        x = concatenate([x, attention_gate(e2, x, nf)])
        x = residual_block(x, nf*2)

        x = Conv3DTranspose(nf, 3, strides=2, padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        x = concatenate([x, attention_gate(e1, x, nf//2)])
        x = residual_block(x, nf)

        # Đầu ra là liều dự đoán
        out = Conv3D(1, kernel_size=1, padding="same")(x)
        out = Activation("relu", name="dose_output")(out)

        # Khởi tạo mô hình
        model = Model(inputs=[ct_input, roi_input], outputs=out, name="EnhancedAttentionResUNet3D")
        model.compile(optimizer=self.gen_optimizer, loss=masked_mae)
        model.summary()
        return model

# ===== HÀM MẤT MẤT (MAE) VỚI MASK =====
def masked_mae(y_true, y_pred):
    dose_true = y_true[..., 0]   # Liều thật
    mask = y_true[..., 1]        # Mask chỉ vùng có GT
    error = tf.abs(dose_true - tf.squeeze(y_pred, axis=-1)) * mask
    return tf.reduce_sum(error) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon())
