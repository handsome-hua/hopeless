import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import nibabel as nib
import os
from io import BytesIO
import random
import zipfile
import shutil

# folder_path = '.\incise_dataset\Train'
# # 创建Zip文件
# zip_file_path = "nii_files.zip"
# with zipfile.ZipFile(zip_file_path, "w") as zf:
#     paths = random.sample(os.listdir(folder_path), k=3)
#     for path in paths:
#         file_path = os.path.join(folder_path, path)
#         for foldername, _, filenames in os.walk(file_path):
#             for filename in filenames:
#                 every_file = os.path.join(foldername, filename)
#                 rel_path = os.path.relpath(every_file, foldername)  # 计算相对路径
#                 zip_info = zipfile.ZipInfo(rel_path)
#                 zf.write(every_file, arcname=zip_info.filename)
# # 读取zip文件为二进制数据
# with open(zip_file_path, 'rb') as f: # 将文件读取为二进制数据就解决了未知文件格式！！
#     bytes = f.read()
# # 显示下载按钮
# st.sidebar.download_button(label="Download NII Files", data=bytes, file_name="nii_files.zip", mime="application/zip")
# # 删除zip文件
# os.remove(zip_file_path)

# 检查session_state中是否存在分割结果（们将其命名为"split_result"）
if "split_result" not in st.session_state:
    # 如果不存在，则初始化一个空值或默认值
    st.session_state.split_result = None

# Nifti 文件（.nii）是一种用于存储医学图像（如脑成像、MRI、CT 等）数据的文件格式，它通常用于神经科学和医学研究中。
# Nifti 文件实际上是基于 Analyze 文件格式的扩展，支持更高的数据精度和更大的数据容量。
# Nifti 文件中的图像数据是三维的，而且可以支持四维及更高维度的图像数据。
# Nifti 文件中的三维图像数据通常包含三个方向（x、y、z），每个方向都有一个像素值矩阵。
# 例如，一个大小为 64x64x30 的 Nifti 文件就包含了 64x64 个像素，每个像素值都对应于空间中的一个位置，并且这些像素沿着 30 个不同的 "切片"（z 方向）排列。
# Nifti 文件中的每个像素值都对应于空间中一个位置的强度或密度，可以用于分析和可视化。
# 因此，可以说 Nifti 文件中的图像数据是三维的，同时也可以扩展到更高的维度
# 定义Dice损失系数
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        # K: tensorflow.keras.backend
        # K.flattern 展平一个向量为 1D 的张量。
        y_true_f = K.flatten(y_true[:, :, :, i])
        y_pred_f = K.flatten(y_pred[:, :, :, i])
        intersection = K.sum(y_true_f * y_pred_f)
        # Dice损失
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        #     K.print_tensor(loss, message='loss value for class {} : '.format(SEGMENT_CLASSES[i]))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
    #    K.print_tensor(total_loss, message=' total dice coef: ')
    return total_loss

# 坏死
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, 1] * y_pred[:, :, :, 1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:, :, :, 1])) + K.sum(K.square(y_pred[:, :, :, 1])) + epsilon)

# 水肿
def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, 2] * y_pred[:, :, :, 2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:, :, :, 2])) + K.sum(K.square(y_pred[:, :, :, 2])) + epsilon)

# 增强肿瘤
def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, 3] * y_pred[:, :, :, 3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:, :, :, 3])) + K.sum(K.square(y_pred[:, :, :, 3])) + epsilon)

# 计算精确度
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# 计算灵敏度
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # clip：元素级的值裁剪；round：以元素方式四舍五入最接近的整数。
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# 计算特异性、特征
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

# 构建模型文件的绝对路径
model_dir = os.path.abspath(os.path.join(os.getcwd(), "incise_dataset"))
model_path = os.path.join(model_dir, "model_x1_1.h5")
# 加载模型
model = load_model(model_path,custom_objects={'dice_coef':dice_coef,'precision':precision,'dice_coef_necrotic':dice_coef_necrotic,'dice_coef_edema':dice_coef_edema,'dice_coef_enhancing':dice_coef_enhancing,'sensitivity':sensitivity,'specificity':specificity},compile=False)
# 设置页面标题和头部
st.title("脑肿瘤图像分割")
st.header("欢迎使用我们的应用:jack_o_lantern:")

# flair.nii 包含的是fluid attenuated inversion recovery（FLAIR）序列的脑部MRI图像数据。
# t1.nii 包含的是磁共振成像（MRI）中的T1加权图像，通常用于结构成像和鉴别诊断。
# t1ce.nii 包含的是T1加权后的小胶质增强图像，它通常用于检测肿瘤细胞和建立分割。
# t2.nii 包含的是磁共振成像（MRI）中的T2加权图像，通常用于检测脑部异常情况，如肿瘤、出血等。¶
# seg.nii 包含的是MRI图像的分割结果，即已经将MRI图像中的不同组织和器官分别标注为不同的标签。

# 加载图像并进行预处理
uploaded_files = st.file_uploader('请上传相关的MRI-nii文件',type=["nii","nii.gz"],accept_multiple_files=True,label_visibility="collapsed")
st.info('Please upload  ***relevant images*** file ed.', icon="ℹ️")
# print(type(uploaded_files)) # list类型
# print(uploaded_files) # 可以将其转换为字符串来判断存在一些字符

def create_or_empty_directory(dir_path):
    # 如果目录不存在，就创建目录
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def save_uploaded_file(uploadedfile):
    """
    保存上传的NIfTI文件到目录下
    """
    # 上传的nii文件没有name属性（通常情况下应该有），您可以尝试使用os模块和时间戳来为每个上传的文件命名
    filename = f"{uploadedfile.name}"
    with open(os.path.join(UPLOAD_DIR, filename), 'wb') as f:

        # uploadedfile.getbuffer()返回一个代表上传文件数据的 bytes 类型对象，可以用于后续处理或存储
        # 注意：getbuffer() 方法可用于获取上传文件的数据，并通过内存缓冲区来避免磁盘访问和 I/O 操作，从而提高数据访问和处理的效率。
        # 但是，由于缓冲区需要占用系统内存空间，因此要特别注意上传文件的大小，以避免内存不足或应用程序崩溃等问题

        f.write(uploadedfile.getbuffer())
    return filename
    # return os.path.join(UPLOAD_DIR, filename)

def save_and_show_file():
    """
    保存上传文件并且展示上传照片名
    """
    expander = st.sidebar.expander("See your file👇")
    # 当expander中没有内容时，它将不会显示在页面上
    with expander:
        st.empty()
    # 指定上传目录和使用示例
    create_or_empty_directory(UPLOAD_DIR)
    if uploaded_files is not None:
        # 处理所有上传的文件
        # expander = st.sidebar.expander("See your file👇")
        for uploaded_file in uploaded_files:
            # 保存上传文件
            filepath = save_uploaded_file(uploaded_file)
            # st.sidebar.success(f"Saved file: {filepath}")
            expander.write(filepath)

# # 定义一个删除文件的函数
# def delete_file(file_path):
#     os.remove(file_path)
#
# def delete_uploaded_file_sametime(file_names):
#     """
#     一个ui和目录下同时删除的代码
#     """
#     if uploaded_files is not None:
#         uploaded_file_names = [file.name for file in uploaded_files]
#         # if set(uploaded_file_names) == set(file_names):
#             # st.success('上传文件成功！')
#         while set(uploaded_file_names) != set(file_names):
#             # 删除目录下多余的文件
#             for file_name in file_names:
#                 if file_name not in uploaded_file_names:
#                     file_path = os.path.join(UPLOAD_DIR, file_name)
#                     # 创建一个新的线程来删除文件
#                     thread = threading.Thread(target=delete_file, args=(file_path,))
#                     thread.start()
#                     # 更新文件列表
#                     file_names.remove(file_name)# 毕竟我是用目录创建的，这个file_names变量变化满意一步，所以在这里我要使用 file_names.remove(file_name)
#             # st.warning('上传文件与目录下的文件不一致，已删除多余的文件！')
#     # else:
#     #     st.warning('请上传文件！')

UPLOAD_DIR = "./uploaded_nii_files"
save_and_show_file()
file_names = os.listdir(UPLOAD_DIR)
# delete_uploaded_file_sametime(file_names)

# 标出相应文件的标识
flair_found = False
t1ce_found = False
seg_found = False
t1_found = False
t2_found = False
flair_data = None
t1ce_data = None
seg_data = None
t1_data = None
t2_data = None
# flair，t1ce，seg,t2等图像预获取
for every in file_names:
    if 'flair' in every:
        filepath_flair = os.path.join(UPLOAD_DIR,every)
        flair_data = nib.load(filepath_flair).get_fdata() # 先加载 Nifti 文件，然后get_fdata获取图像数据
        flair_found = True
        # print(flair_data.shape) # (240, 240, 155)
        continue
    if 't1ce' in every:
        filepath_t1ce = os.path.join(UPLOAD_DIR, every)
        t1ce_data = nib.load(filepath_t1ce).get_fdata()
        t1ce_found = True
        continue
    if 'seg' in every:
        filepath_seg = os.path.join(UPLOAD_DIR, every)
        seg_data = nib.load(filepath_seg).get_fdata()
        seg_found = True
        continue
    if 't1' in every:
        filepath_t1 = os.path.join(UPLOAD_DIR, every)
        t1_data = nib.load(filepath_t1).get_fdata()
        t1_found = True
        continue
    if 't2' in every:
        filepath_t2 = os.path.join(UPLOAD_DIR, every)
        t2_data = nib.load(filepath_t2).get_fdata()
        t2_found = True

shutil.rmtree('uploaded_nii_files')
slice_w = 25
# tab1, tab2, tab3, tab4, tab5 = st.tabs(["Flair-Img", "T1-Img", "T1ce-Img", "T2-Img", "Mask-Img"])
# 创建显示图像的函数
@st.cache_data(show_spinner=False)
def show_flair(flair_data):
    try:
        st.header("Image flair")
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
        axs[1].imshow(flair_data[:, :, flair_data.shape[0] // 2 - slice_w], cmap='gray')
        for ax in axs:
            ax.axis('off')
        st.pyplot(fig)
    except Exception as e:
        pass
        # st.error(f'Error:{e}', icon="🚨")

@st.cache_data(show_spinner=False)
def show_t1(t1_data):
    try:
        st.header("Image t1")
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
        axs[1].imshow(t1_data[:, :, t1_data.shape[0] // 2 - slice_w], cmap='gray')
        for ax in axs:
            ax.axis('off')
        st.pyplot(fig)
    except Exception as e:
        pass
        # st.error(f'Error:{e}', icon="🚨")


@st.cache_data(show_spinner=False)
def show_t1ce(t1ce_data):
    try:
        st.header("Image t1ce")
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
        axs[1].imshow(t1ce_data[:,:,t1ce_data.shape[0] // 2 - slice_w], cmap='gray')
        for ax in axs:
            ax.axis('off')
        st.pyplot(fig)
    except Exception as e:
        pass
        # st.error(f'Error:{e}', icon="🚨")

@st.cache_data(show_spinner=False)
def show_t2(t2_data):
    try:
        st.header("Image t2")
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
        axs[1].imshow(t2_data[:, :, t2_data.shape[0] // 2 - slice_w], cmap='gray')
        for ax in axs:
            ax.axis('off')
        st.pyplot(fig)
    except Exception as e:
        pass
        # st.error(f'Error:{e}', icon="🚨")

@st.cache_data(show_spinner=False)
def show_mask(seg_data):
    try:
        st.header("Mask")
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
        axs[1].imshow(seg_data[:, :, seg_data.shape[0] // 2 - slice_w])
        for ax in axs:
            ax.axis('off')
        st.pyplot(fig)
    except Exception as e:
        # st.error(f'Error:{e}', icon="🚨")
        pass

tab = st.sidebar.radio("Choose Image Type", ["Flair-Img", "T1-Img", "T1ce-Img", "T2-Img", "Mask-Img"])
if tab == "Flair-Img":
    show_flair(flair_data)
elif tab == "T1-Img":
    show_t1(t1_data)
elif tab == "T1ce-Img":
    show_t1ce(t1ce_data)
elif tab == "T2-Img":
    show_t2(t2_data)
else:
    show_mask(seg_data)

IMG_SIZE = 128 # 规定输入图像的大小
start_slice = 60 #
# 给不同区域定标签
SEGMENT_CLASSES = {
    0 : 'NOT tumor',  # 正常部分
    1 : 'NECROTIC/CORE', # 坏疽 NON-ENHANCING tumor CORE
    2 : 'EDEMA',  # 水肿
    3 : 'ENHANCING' # 增强肿瘤 原本的标签是 4 -> 这里将它改成了3，使标签连续，更好编写代码逻辑
}

# 每个样例有 155 张切片图
# 如果需要读取一张医学图像数据的连续 100 层，且想从第 22 层开始读取
# 则可以将 VOLUME_SLICES 设置为 100，VOLUME_START_AT 设置为 22
VOLUME_SLICES = 100
VOLUME_START_AT = 22 # first slice of volume that we will include

@st.cache_resource(show_spinner="Processing segmented images...") # 自定义微调器
# 默认情况下，当缓存函数正在运行时，Streamlight会在应用程序中显示一个小的加载微调器。您可以使用show_spinner参数轻松修改它
def seg_start_image(flair_data,t1ce_data,seg_data):
    # placeholder = st.empty()
    # placeholder.write("⏳ The program is executing")
    try:
        X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))  # (100,128,128,2)
        for j in range(VOLUME_SLICES):
            # 从22-122
            X[j, :, :, 0] = cv2.resize(flair_data[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
            X[j, :, :, 1] = cv2.resize(t1ce_data[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
            # y[j,:,:] = cv2.resize(seg[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))

        p = model.predict(X / np.max(X), verbose=1) # (100, 128, 128, 4)
        core = p[:, :, :, 1]
        edema = p[:, :, :, 2]
        enhancing = p[:, :, :, 3]

        fig, axarr = plt.subplots(1, 6, figsize =(18, 20))

        for i in range(6):  # for each image, add brain background
            axarr[i].imshow(cv2.resize(flair_data[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)),
                                cmap="gray", interpolation='none')

        for ax in axarr.flatten():
            ax.axis('off')
        axarr[0].imshow(cv2.resize(flair_data[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
        axarr[0].title.set_text('Original Image Flair')
        axarr[1].imshow(p[start_slice, :, :, 1:4], cmap="Reds", interpolation='none', alpha=0.3)
        axarr[1].title.set_text('ALL CLASSES')
        axarr[2].imshow(edema[start_slice, :, :], cmap="OrRd", interpolation='none', alpha=0.3)
        axarr[2].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
        axarr[3].imshow(core[start_slice, :, ], cmap="OrRd", interpolation='none', alpha=0.3)
        axarr[3].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
        axarr[4].imshow(enhancing[start_slice, :, ], cmap="OrRd", interpolation='none', alpha=0.3)
        axarr[4].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')

        axarr[0].title.set_fontsize(14)
        axarr[1].title.set_fontsize(14)
        axarr[2].title.set_fontsize(14)
        axarr[3].title.set_fontsize(14)
        axarr[4].title.set_fontsize(14)

        if seg_data is not None and seg_found:
            curr_gt = cv2.resize(seg_data[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE),
                                 interpolation=cv2.INTER_NEAREST)
            axarr[5].imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3)  # ,alpha=0.3,cmap='Reds'
            axarr[5].title.set_text('Ground Truth')
            axarr[5].title.set_fontsize(14)
        else:
            # 创建一个空白的全零数组即全黑
            blank_image = np.zeros((IMG_SIZE, IMG_SIZE))
            axarr[5].imshow(blank_image,cmap="gray", interpolation='none')
            axarr[5].title.set_text('NONE-Ground Truth')
            axarr[5].title.set_fontsize(14)
        # 用 st.pyplot() 函数显示图像
        # st.pyplot(fig) # 这种不能够持久化

        # placeholder.empty()

        # 这样，在每次与应用程序交互时，Streamlit会重新运行脚本代码，但st.session_state.split_result将保留上一次计算的分割图片。
        # 当有新的分割结果时，页面上显示的图片也会更新。
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight") # 用 fig.savefig() 函数保存图像到字节流(BytesIO)对象
        buf.seek(0) # 用于将文件指针移动到文件开头的方法,这样就可以重新读取整个文件了。
        # 将子图存储在session_state.split_result中
        st.session_state.split_result = buf

    # 如果报错也可能只存在没有上传相应的图片的问题了，这个问题肯定显而易见，所以在except下可以直接使用pass。
    except ValueError as ve:
        st.warning(str(ve),icon="⚠️")
        st.session_state.split_result = None  # 出现了错误，将状态清零
    except Exception:
        st.warning('Warning: Flair,T1CE are required to exist at the same time',icon="⚠️")
        st.session_state.split_result = None # 出现了错误，将状态清零

# 进行脑肿瘤的分类和分割
if st.button('**分割按钮**',key='seg_zhongliu',type="primary"):
    st.cache_resource.clear()  # 强制清除缓存
    seg_start_image(flair_data,t1ce_data,seg_data)  # seg_start_image(flair_data,t1ce_data,seg_data)是一个返回分割结果的函数

# 展示分割结果（如果有的话）
if st.session_state.split_result is not None:
    st.image(st.session_state.split_result)

    # st.write(type(st.session_state.split_result[0]))
    # for i in st.session_state.split_result:
    #     st.image(i)
