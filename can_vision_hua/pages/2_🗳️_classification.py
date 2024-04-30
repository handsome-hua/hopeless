import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from tensorflow.keras.models import load_model
import random
import os
import zipfile

# 有一点不同的是，我采用的是pages多页面，它的当前目录的路径与pythonPython 的标准输出中打印当前工作目录的路径是不一样的！
# 也就是说通过streamlit run hello.py 运行的页面，其当前工作目录是跳过pages
# print(os.getcwd())
# st.write(os.getcwd())

IMAGE_SIZE = 128
# 定义相应标签
FENLEIMENT_CLASSES = {
    0 : 'Glioma',  #神经胶质瘤
    1 : 'Meningioma', # 脑膜瘤
    2 : 'Notumor',  # 正常
    3 : 'Pituitary' # 脑垂瘤
}

# 数据图像增强
def augment_image(image):
    image = Image.fromarray(np.uint8(image)) # PIL 库中的图像对象要求像素值的数据类型为 uint8 类型，即无符号 8 位整数类型。
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8,1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8,1.2))
    # 注意：归一化不能去掉，模型训练是就是这样的
    image = np.array(image)/255.0 # 转换后的数组可以使用 NumPy 库中的函数进行处理和分析 图像归一化可以有助于在训练神经网络时提高性能和收敛速度。
    # 当结合两组特征时,如果两者的scale和norm相差很大的话，则大的一方容易吞掉小的一方,如果使用归一化，则能较好的避免该问题，且训练更稳定。
    return image

@st.cache_resource(show_spinner="Running...")
def open_images(current_file):
    '''
    Given a current_file to images, this function returns the images as arrays (after augmenting them)
    '''
    images = [] # images的作用是为了配合模型的维度
    image = Image.open(current_file)
    # 将图像大小调整为128x128
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # 如果图像是灰度图，将其转换为RGB
    if image.mode == 'L':
        image = image.convert('RGB')
    # 图像亮度对比度增强
    image = augment_image(image)
    # 保存维度与模型expect一致
    images.append(image)
    # 转为np处理
    images = np.array(images)

    # 模型预测
    pred = your_model.predict(images)
    pred = np.argmax(pred, axis=-1)

    name = FENLEIMENT_CLASSES[pred[0]]  # 因为是单个文件，所以pred只有单个变量数字
    return name,image
# temp_folders = [r'.\picturewithfenlei\Training\glioma',
#                 r'.\picturewithfenlei\Training\meningioma',
#                 r'.\picturewithfenlei\Training\notumor',
#                 r'.\picturewithfenlei\Training\pituitary']
# # 创建Zip文件
# zip_file_path = "images.zip"
# with zipfile.ZipFile(zip_file_path, "w") as zf:
#     for temp_folder in temp_folders:
#         for foldername, _, filenames in os.walk(temp_folder):
#             filenames = random.sample(filenames, k=3)
#             for filename in filenames:
#                 file_path = os.path.join(foldername, filename)
#                 rel_path = os.path.relpath(file_path, temp_folder) # 计算相对路径
#                 zip_info = zipfile.ZipInfo(rel_path) # zipfile.ZipInfo()函数使用这个相对路径来创建一个与之关联的zipfile.ZipInfo对象,zipfile.ZipInfo对象包含了与文件或目录相关的各种信息.
#                 zf.write(file_path, arcname=zip_info.filename)

# # 读取zip文件为二进制数据
# with open(zip_file_path, 'rb') as f: # 将文件读取为二进制数据就解决了未知文件格式！！
#     bytes = f.read()

# # 在Streamlit的download_button函数中，data参数需要的是一个字节流（binary stream）或者文本数据，而不是文件路径。
# # 当你直接传入一个文件路径时，Streamlit并不能正确地将文件内容读入到字节流中，因此在下载后的文件会出现格式错误。
# # 当你使用open函数以二进制模式（'rb'）打开并读取文件后，你得到的是一个字节流，这个字节流是可以被download_button函数正确处理的，因此在下载后的文件就不会有格式错误。
# # 总的来说，这是因为Streamlit的download_button函数设计如此，它需要的是文件的内容（以字节流或文本数据的形式），而不是文件的路径

# # 显示下载按钮
# st.sidebar.download_button(
#     label="Download Images",
#     data=bytes,
#     file_name="images.zip",
#     mime="application/zip"
# )
# # 删除zip文件
# os.remove(zip_file_path)

# 构建模型文件的绝对路径
model_dir = os.path.abspath(os.path.join(os.getcwd(), "picturewithfenlei"))
model_path = os.path.join(model_dir, "model_fenlei.h5")
# 加载模型
your_model = load_model(model_path)
# 设置页面标题和头部
st.title("脑肿瘤图像分类")
st.header("欢迎使用我们的应用:hugging_face:")

# 允许用户上传多个文件
uploaded_files = st.file_uploader("请上传您的脑肿瘤分类图像（支持多个）:file_folder:", type=["jpg", "jpeg", "png"], accept_multiple_files=True,label_visibility="collapsed")
st.info('Upload relevant images to showcase the results:point_up_2:',icon="ℹ️")

# 获取所有上传文件的名称
names = [file.name for file in uploaded_files]

# 在侧边栏中创建一个选择框，列出所有上传的文件名
selected_file = st.sidebar.selectbox(f"请选择一个文件：({len(names)} Files)", names)
# 单个图像的展示
if uploaded_files:

    # 获取当前选定的文件名对应的File对象
    current_file = next(file for file in uploaded_files if file.name == selected_file) # 从上传的文件列表中找到指定名称的文件对象。如果找到了该文件对象，则返回该文件对象

    name, _ = open_images(current_file)

    # 将展示的图像放大两倍
    image = Image.open(current_file)
    # 将图像大小调整为128x128
    image = image.resize((IMAGE_SIZE*2, IMAGE_SIZE*2))
    # 如果图像是灰度图，将其转换为RGB
    if image.mode == 'L':
        image = image.convert('RGB')
    # 图像亮度对比度增强
    image = augment_image(image)

    st.image(image, caption=(f'{selected_file[:-4]}👉{name}'))
    # st.write(f'{selected_file[:-4]}👉:red[{name}]')
    # st.write('%s👉:red[%s]'%(selected_file[:-4],name))

# 进行图像分类
unique_labels = ['Glioma', 'Meningioma', 'Notumor', 'Pituitary']
tab1 ,tab2 ,tab3 ,tab4 = st.tabs(unique_labels)
glioma_images = []
glioma_names = []
meningioma_images = []
meningioma_names = []
notumor_images = []
notumor_names = []
pituitary_images = []
pituitary_names = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        name_catergory,image = open_images(uploaded_file) # 获取种类及其增强图像
        if name_catergory == 'Glioma':
            glioma_images.append(image)
            glioma_names.append(uploaded_file.name)
        elif name_catergory == 'Meningioma':
            meningioma_images.append(image)
            meningioma_names.append(uploaded_file.name)
        elif name_catergory == 'Pituitary':
            pituitary_images.append(image)
            pituitary_names.append(uploaded_file.name)
        else:
            notumor_images.append(image)
            notumor_names.append(uploaded_file.name)

glioma_images = np.array(glioma_images)
meningioma_images = np.array(meningioma_images)
notumor_images = np.array(notumor_images)
pituitary_images = np.array(pituitary_images)

# st.write(notumor_images.shape)

cloumns = 4  # 4列
# row_max = max([len(glioma_images),len(meningioma_images),len(pituitary_images),len(notumor_images)]) # 选取最多图片，尽量保证各个选项卡图像分布一致

# rows = row_max // cloumns # 根据图像的数量来设置多少行（最小）
# if row_max % cloumns != 0: # 据图像的数量来设置多少行（最大）
#     rows += 1


def plot_images(images, names):
    cloumns = 4
    rows = len(images) // cloumns # 根据图像的数量来设置多少行（最小
    if len(images) % cloumns != 0: # 据图像的数量来设置多少行（最大）
        rows += 1

    fig_width = cloumns * 4
    fig_height = rows * 4

    fig, axarr = plt.subplots(rows, cloumns, figsize=(fig_width, fig_height))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)  # 设置子图之间的水平,垂直间距

    nums = 0
    for row in range(rows):
        for column in range(cloumns):
            if nums >= len(images):
                if rows == 1 or cloumns == 1:
                    axarr[nums].axis('off')
                else:
                    axarr[row][column].axis('off')
                nums += 1
                continue

            if rows == 1 or cloumns == 1:
                ax = axarr[nums]
            else:
                ax = axarr[row][column]

            ax.imshow(images[nums])
            ax.axis('off')
            ax.title.set_text(names[nums])
            ax.title.set_fontsize(12)
            nums += 1

    return fig

# # 注意图片的名称也加上，这样多个图片中，可以更好的认识是哪张图片了
# with tab1:
#     st.caption(f'Here are {len(glioma_images)} images of Glioma')
#     glioma_check = st.sidebar.checkbox('Show glioma', key='glioma')
#     if glioma_check:
#         try:
#             fig = plot_images(glioma_images, glioma_names)
#             st.pyplot(fig)
#         except Exception as e:
#             st.warning(f'Error:{e}', icon="⚠️")
#             # st.warning('Error: {}'.format('No glioma image exists'), icon="⚠️")

def image_list(images,names):
    nums = 0
    for image_ in images:
        cols[nums % cloumns].image(image_, caption=names[nums], use_column_width=True)
        nums += 1

# 将st.columns和st.image结合，即每一列轮次各画一个图，直到画完。
with tab1:
    try:
        st.caption(f'Here are {len(glioma_images)} images of Glioma')
        glioma_check = st.sidebar.checkbox('Show glioma', key='glioma')
        col1, col2, col3 ,col4 = st.columns(cloumns)
        cols = [col1,col2,col3,col4]
        if glioma_check:
            image_list(glioma_images,glioma_names)
    except Exception as e:
        st.warning(f'Error:{e}', icon="⚠️")

with tab2:
    try:
        st.caption(f'Here are {len(meningioma_images)} images of Meningioma')
        meningioma_check = st.sidebar.checkbox('Show meningioma', key='meningioma')
        col1, col2, col3 ,col4 = st.columns(cloumns)
        cols = [col1,col2,col3,col4]
        if meningioma_check:
            image_list(meningioma_images,meningioma_names)
    except Exception as e:
        st.warning(f'Error:{e}', icon="⚠️")

with tab3:
    try:
        st.caption(f'Here are {len(notumor_images)} images of Notumor')
        notumor_check = st.sidebar.checkbox('Show notumor', key='notumor')
        col1, col2, col3 ,col4 = st.columns(cloumns)
        cols = [col1,col2,col3,col4]
        if notumor_check:
            image_list(notumor_images,notumor_names)
    except Exception as e:
        st.warning(f'Error:{e}', icon="⚠️")

with tab4:
    try:
        st.caption(f'Here are {len(pituitary_images)} images of Pituitary')
        pituitary_check = st.sidebar.checkbox('Show pituitary', key='pituitary')
        col1, col2, col3 ,col4 = st.columns(cloumns)
        cols = [col1,col2,col3,col4]
        if pituitary_check:
            image_list(pituitary_images,pituitary_names)
    except Exception as e:
        st.warning(f'Error:{e}', icon="⚠️")
