import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ----------------------------------------------------
# 1. データの準備と前処理
# ----------------------------------------------------
# データをロード（手書き数字の画像データセット）
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# 今回はkerasのデータセットを使用するためロードする
# テストデータと訓練データを分ける

# ----------------------------------------------------
# 2. 学習データの可視化
# ----------------------------------------------------
print("読み込んだ学習データの一部を可視化します...")

# 5x5のグリッドで25枚の画像を表示する準備
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    # 軸のメモリは不要なので非表示に
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # x_train[i]は28x28の画像データ
    # cmap=plt.cm.binaryで白黒表示にする
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    # 画像の下に正解ラベル(y_train[i])を表示
    plt.xlabel(f"Label: {y_train[i]}")

plt.suptitle("MNIST Training Data Samples", fontsize=16) # 全体のタイトル
plt.show()

# データを0〜1の範囲に正規化（前処理）
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# 色のピクセルは黒0~白255までだから255で正規化

# 画像データを1次元のベクトルに変換（平坦化）
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
#この画像データは28×28ピクセルの2次元だから28×28=784の1次元にする

# ----------------------------------------------------
# 3. モデルの構築
# ----------------------------------------------------
# Sequentialモデル（層を積み重ねるモデル）を定義
model = keras.Sequential([
    # 第1層：隠れ層（512個のニューロン）
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    # 第2層：隠れ層（256個のニューロン）
    keras.layers.Dense(256, activation='relu'),
    # 出力層：0～9の10クラス分類なので10個のニューロン
    keras.layers.Dense(10, activation='softmax')
])
#2のべき乗(512，256など)はハードウェアの効率上好まれる数字である。他の数値でも学習は可。
# 一般的にニューロンは層が深くなるほど減らす。

# ----------------------------------------------------
# 4. モデルの学習
# ----------------------------------------------------
# モデルのコンパイル（学習方法の設定）
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# コンパイルは最適化アルゴリズム、損失関数、評価指標を設定する。

# モデルの学習（訓練データを使って重みを最適化）
model.fit(x_train, y_train, epochs=5, batch_size=32)

# ----------------------------------------------------
# 5. モデルの評価
# ----------------------------------------------------
# 訓練していない評価用データを使ってモデルをテスト
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("-" * 30)
print(f"テストデータの損失 (Loss): {loss:.4f}")
print(f"テストデータの正解率 (Accuracy): {accuracy:.4f}")
print("-" * 30)
# エポッチは訓練データセットを何周させるかの値
# バッチサイズは大きいと学習時間が短いが汎化性能化が落ちる。小さいと計算効率や学習が安定しない。

# ----------------------------------------------------
# 6. モデル評価の可視化
# ----------------------------------------------------

print("モデルの評価結果を可視化します...")

# テストデータに対する予測を実行
y_pred_probs = model.predict(x_test)
y_pred_labels = np.argmax(y_pred_probs, axis=1) # 確率が最も高いクラスを予測ラベルとする

# 混同行列を計算して表として出力
# scikit-learnのconfusion_matrixを使って、正解ラベル(y_test)と予測ラベル(y_pred_labels)を比較
cm = confusion_matrix(y_test, y_pred_labels)

# 混同行列をPandasのDataFrameに変換（行と列に見出しをつけるため）
cm_df = pd.DataFrame(cm,
                     index=[f'正解:{i}' for i in range(10)],
                     columns=[f'予測:{i}' for i in range(10)])

print("\n--- 混同行列 (Confusion Matrix) ---")
print("行: 正解ラベル (Actual), 列: 予測ラベル (Predicted)")
print(cm_df)
print("-" * 40)

# 混同行列をヒートマップで可視化
# 表を色分けして視覚的に分かりやすくする
plt.figure(figsize=(10, 8)) # グラフのサイズを指定
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
# annot=True: 数値をセルに表示
# fmt='d': 数値を整数で表示
# cmap='Blues': カラーマップを青系に

plt.title('Confusion Matrix Heatmap')
plt.ylabel('Actual Label (正解ラベル)')
plt.xlabel('Predicted Label (予測ラベル)')
plt.show()