# 必要なライブラリをインポート
#streamlit run main.pyが実行
#command Cで終了
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import models
from torchvision import transforms
import pickle

# タイトルとテキストを記入
st.title('鳥の種類判別')

model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 200)
model.load_state_dict(torch.load("weights/bird_classifier_model.pth", map_location='cpu'))
model.eval()



st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("鳥の種類")
st.sidebar.write("ResNetを使用")
st.sidebar.write("")



# 画像のソースを選択するラジオボタンの追加
img_source = st.sidebar.radio("画像のソースを選択してください。", ["画像をアップロード"])
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
    st.session_state.uploaded_image = img_file



def initialize_session_state():
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None




#リセットボタンの作成
reset = st.sidebar.button('画像をリセット')
if reset:
    st.session_state.uploaded_image = None
    img_file = None
    initialize_session_state()

# 画像の前処理変換、予測には検証用データの変換を適用
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


with open('label_to_name.pkl', 'rb') as f:
    class_names = pickle.load(f)


# 予測関数の定義
def predict(image):
    # 画像の前処理
    preprocess = data_transforms['val']
    image = preprocess(image).unsqueeze(0)  # バッチの次元を追加

    # 予測の実行
    with torch.no_grad():
        outputs = model(image)
        probs, indices = torch.topk(outputs, 5)  # トップ5の確率とインデックス
        probs = torch.nn.functional.softmax(probs, dim=1)[0] * 100  # Softmaxしてパーセンテージに変換
        labels = [class_names[int(idx.item())] for idx in indices[0]]

    return list(zip(labels, probs.tolist()))


if img_file is not None:
    img = Image.open(img_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    results = predict(img)  # 予測関数から結果を取得

    # 結果の表示
    st.subheader("判定結果")
    n_top = 5  # 確率が高い順に5位まで返す
    for result in results[:n_top]:
         st.write(str(round(result[1], 2)) + "%の確率で" + result[0] + "です。")

    # 横棒グラフの表示
    bar_labels = [result[0] for result in results[:n_top]]
    bar_probs = [result[1] for result in results[:n_top]]

    fig, ax = plt.subplots(figsize=(10, 7))
    # x軸のラベルに「確率」と表示
    ax.set_xlabel('確率')
    # 表題に「予測確率」と表示
    ax.set_title('予測確率')

    y_pos = range(len(bar_labels))
    ax.barh(y_pos, bar_probs, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_labels)
    ax.invert_yaxis()  # ラベルを上から表示する


    plt.tight_layout()
    st.pyplot(fig)



