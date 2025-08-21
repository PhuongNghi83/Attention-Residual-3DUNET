'''# Attention U-Net for Radiotherapy Dose Prediction (TensorFlow 2.13 + Python 3.11 Compatible)

# Cài đặt thư viện:
# pip install tensorflow==2.13.0 numpy pandas matplotlib scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, backend as K
from sklearn.model_selection import train_test_split

# Load dữ liệu
combined_X = np.load('saved_data/combined_X.npy')
combined_Y = np.load('saved_data/combined_Y.npy')

X_train, X_test, Y_train, Y_test = train_test_split(combined_X, combined_Y, test_size=0.2, random_state=1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)

img_height, img_width, img_depth, img_channels = X_train.shape[1:]
input_shape = (img_height, img_width, img_depth, img_channels)

# Các hàm khối U-Net + attention
def conv_block(x, size, dropout):
    x = layers.Conv3D(size, 3, padding="same", activation='relu', kernel_initializer='he_uniform')(x)
    x = layers.Conv3D(size, 3, padding="same", activation='relu', kernel_initializer='he_uniform')(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x

def gating_signal(input, out_size):
    return layers.Conv3D(out_size, 1, padding='same', activation='relu', kernel_initializer='he_uniform')(input)

def repeat_elem(tensor, rep):
    return tf.repeat(tensor, rep, axis=-1)

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    theta_x = layers.Conv3D(inter_shape, 2, strides=2, padding='same', kernel_initializer='he_uniform')(x)
    phi_g = layers.Conv3D(inter_shape, 1, padding='same', kernel_initializer='he_uniform')(gating)

    shape_theta_x = K.int_shape(theta_x)
    upsample_g = layers.Conv3DTranspose(inter_shape, 3,
        strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2], shape_theta_x[3] // shape_g[3]),
        padding='same', kernel_initializer='he_uniform')(phi_g)

    xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(xg)
    psi = layers.Conv3D(1, 1, padding='same', kernel_initializer='he_uniform')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)

    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling3D(
        size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3])
    )(sigmoid_xg)

    upsample_psi = repeat_elem(upsample_psi, shape_x[4])
    y = layers.multiply([upsample_psi, x])
    result = layers.Conv3D(shape_x[4], 1, padding='same', kernel_initializer='he_uniform')(y)
    return result

def Attention_UNet_3D_Model(input_shape):
    f = 16
    inputs = Input(input_shape)

    # Downsampling
    c1 = conv_block(inputs, f, 0.1); p1 = MaxPooling3D(2)(c1)
    c2 = conv_block(p1, f*2, 0.15); p2 = MaxPooling3D(2)(c2)
    c3 = conv_block(p2, f*4, 0.2); p3 = MaxPooling3D(2)(c3)
    c4 = conv_block(p3, f*8, 0.25); p4 = MaxPooling3D(2)(c4)
    c5 = conv_block(p4, f*16, 0.3)

    # Upsampling + attention
    g4 = gating_signal(c5, f*8)
    att4 = attention_block(c4, g4, f*8)
    u4 = concatenate([layers.UpSampling3D(2)(c5), att4])
    c6 = conv_block(u4, f*8, 0.25)

    g3 = gating_signal(c6, f*4)
    att3 = attention_block(c3, g3, f*4)
    u3 = concatenate([layers.UpSampling3D(2)(c6), att3])
    c7 = conv_block(u3, f*4, 0.2)

    g2 = gating_signal(c7, f*2)
    att2 = attention_block(c2, g2, f*2)
    u2 = concatenate([layers.UpSampling3D(2)(c7), att2])
    c8 = conv_block(u2, f*2, 0.15)

    g1 = gating_signal(c8, f)
    att1 = attention_block(c1, g1, f)
    u1 = concatenate([layers.UpSampling3D(2)(c8), att1])
    c9 = conv_block(u1, f, 0.1)

    outputs = Conv3D(1, 1, activation='linear')(c9)
    return Model(inputs, outputs)

# Train model
model = Attention_UNet_3D_Model(input_shape)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
model.fit(X_train, Y_train,
          validation_data=(X_val, Y_val),
          epochs=50,
          batch_size=2,
          callbacks=[early_stop])

model.save('saved_model/att_unet_compatible_py311.h5')
print("✅ Model trained and saved successfully.")

# ================================
# PHẦN DỰ ĐOÁN + VẼ ẢNH
# ================================

Y_pred = model.predict(X_test, batch_size=2)
Y_pred[Y_pred < 0] = 0

Y_pred = np.squeeze(Y_pred) * 70
Y_true = np.squeeze(Y_test) * 70

image_index = 0
slice_index = 32

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(Y_true[image_index, :, :, slice_index], cmap='jet')
axs[0].set_title("Ground Truth"); axs[0].axis('off')

axs[1].imshow(Y_pred[image_index, :, :, slice_index], cmap='jet')
axs[1].set_title("Predicted"); axs[1].axis('off')

axs[2].imshow(Y_pred[image_index, :, :, slice_index] - Y_true[image_index, :, :, slice_index], cmap='bwr')
axs[2].set_title("Residual"); axs[2].axis('off')

plt.tight_layout()
plt.show()'''
# Attention U-Net for Radiotherapy Dose Prediction (TensorFlow 2.13 + Python 3.11)

# ⚠️ Cài đặt:
# pip install tensorflow==2.13.0 numpy pandas matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, concatenate, UpSampling3D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, backend as K
import os

# ⚙️ Load dữ liệu
X = np.load("saved_data/combined_X.npy")
Y = np.load("saved_data/combined_Y.npy")

# ✅ Kiểm tra dữ liệu
assert not np.isnan(X).any(), "❌ X có NaN"
assert not np.isnan(Y).any(), "❌ Y có NaN"
assert X.shape[1:] == (128, 128, 128, 10), "❌ Sai input shape"
assert Y.shape[1:] == (128, 128, 128), "❌ Sai label shape"

# ✅ Dùng toàn bộ dữ liệu nếu chỉ có 1 mẫu
X_train = X
Y_train = Y

# 🧱 Khối U-Net + Attention
def conv_block(x, size, dropout):
    x = layers.Conv3D(size, 3, padding="same", activation='relu', kernel_initializer='he_uniform')(x)
    x = layers.Conv3D(size, 3, padding="same", activation='relu', kernel_initializer='he_uniform')(x)
    return layers.Dropout(dropout)(x) if dropout > 0 else x

def gating_signal(x, out_size):
    return layers.Conv3D(out_size, 1, padding='same', activation='relu', kernel_initializer='he_uniform')(x)

def repeat_elem(tensor, rep):
    return tf.repeat(tensor, rep, axis=-1)

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)
    theta_x = layers.Conv3D(inter_shape, 2, strides=2, padding='same')(x)
    phi_g = layers.Conv3D(inter_shape, 1, padding='same')(gating)

    shape_theta_x = K.int_shape(theta_x)
    upsample_g = layers.Conv3DTranspose(inter_shape, 3,
        strides=(shape_theta_x[1] // shape_g[1],
                 shape_theta_x[2] // shape_g[2],
                 shape_theta_x[3] // shape_g[3]),
        padding='same')(phi_g)

    xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(xg)
    psi = layers.Conv3D(1, 1, padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)

    upsample_psi = layers.UpSampling3D(size=(
        shape_x[1] // sigmoid_xg.shape[1],
        shape_x[2] // sigmoid_xg.shape[2],
        shape_x[3] // sigmoid_xg.shape[3]))(sigmoid_xg)

    upsample_psi = repeat_elem(upsample_psi, shape_x[4])
    y = layers.multiply([upsample_psi, x])
    return layers.Conv3D(shape_x[4], 1, padding='same')(y)

def Attention_UNet_3D(input_shape):
    f = 16
    inputs = Input(input_shape)

    c1 = conv_block(inputs, f, 0.1); p1 = MaxPooling3D(2)(c1)
    c2 = conv_block(p1, f*2, 0.15); p2 = MaxPooling3D(2)(c2)
    c3 = conv_block(p2, f*4, 0.2); p3 = MaxPooling3D(2)(c3)
    c4 = conv_block(p3, f*8, 0.25); p4 = MaxPooling3D(2)(c4)
    c5 = conv_block(p4, f*16, 0.3)

    g4 = gating_signal(c5, f*8)
    att4 = attention_block(c4, g4, f*8)
    u4 = concatenate([UpSampling3D(2)(c5), att4])
    c6 = conv_block(u4, f*8, 0.25)

    g3 = gating_signal(c6, f*4)
    att3 = attention_block(c3, g3, f*4)
    u3 = concatenate([UpSampling3D(2)(c6), att3])
    c7 = conv_block(u3, f*4, 0.2)

    g2 = gating_signal(c7, f*2)
    att2 = attention_block(c2, g2, f*2)
    u2 = concatenate([UpSampling3D(2)(c7), att2])
    c8 = conv_block(u2, f*2, 0.15)

    g1 = gating_signal(c8, f)
    att1 = attention_block(c1, g1, f)
    u1 = concatenate([UpSampling3D(2)(c8), att1])
    c9 = conv_block(u1, f, 0.1)

    output = Conv3D(1, 1, activation='linear')(c9)
    return Model(inputs, output)

# 🧠 Khởi tạo & huấn luyện mô hình
model = Attention_UNet_3D(input_shape=(128,128,128,10))
model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
model.fit(X_train, Y_train, batch_size=1, epochs=2, callbacks=[early_stop])

# 💾 Lưu mô hình
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/att_unet_compatible_py311.h5")
print("✅ Mô hình đã được huấn luyện và lưu xong.")

# ========================
# 🔍 DỰ ĐOÁN & VẼ KẾT QUẢ
# ========================
Y_pred = model.predict(X_train)
Y_pred[Y_pred < 0] = 0

# scale ngược lại Gy (nếu cần)
Y_pred = np.squeeze(Y_pred) * 70
Y_true = np.squeeze(Y_train) * 70

image_index = 0
slice_index = 64

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(Y_true[:, :, slice_index], cmap='jet')
axs[0].set_title("Ground Truth"); axs[0].axis('off')

axs[1].imshow(Y_pred[:, :, slice_index], cmap='jet')
axs[1].set_title("Predicted"); axs[1].axis('off')

axs[2].imshow(Y_pred[:, :, slice_index] - Y_true[:, :, slice_index], cmap='bwr')
axs[2].set_title("Residual"); axs[2].axis('off')

plt.tight_layout()
plt.show()
