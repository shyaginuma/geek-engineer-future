---
title: "4GM本（Approaching (Almost) Any Machine Learning Problem）を読んだ感想を話しました！"
date: 2020-07-28T18:40:11+09:00
draft: false
---

Podcastは[こちら](https://anchor.fm/geek-engineer-future/)から🎵

👉質問コメントは[質問箱](https://peing.net/ja/04affd1e18a05d/message) or [Twitter](https://twitter.com/)のハッシュタグ[`#geek_engineer`](https://twitter.com/search?q=%23geek_engineer)にてお待ちしております📮

## Summary

通称「4GM本」と呼ばれている[Approaching (Almost) Any Machine Learning Problem](https://www.amazon.co.jp/Approaching-Almost-Machine-Learning-Problem/dp/8269211508)を読んだので、その感想や学びなどをお話しました！

## どんな本？

kaggleで4つのGMの称号を持つ[Abhishek Thakur 氏](https://www.kaggle.com/abhishek)が書いた本。
構成は以下の通り（目次）

```
・Setting up your working environment
・Supervised vs unsupervised learning
・Cross-validation
・Evaluation metrics
・Arranging machine learning projects
・Approaching categorical variables
・Feature engineering
・Feature Selection
・Hyperparameter optimization
・Approaching image classification & segmentation
・Approaching text classification/regression
・Approaching ensembling and stacking
・Approaching reproducible code & model serving
```

コードスニペットも豊富で、kaggleでよく話題になるCross-validationやFeature engineeringといった内容から、Approaching reproducible code & model servingといった実運用を見据えた章もあり、とても勉強になる本だった。（DNNはPyTorchによるスニペット有）

## 他のkaggle書籍との棲み分け
個人的には下記順番で学習するのが良いと思った。

1. [PythonではじめるKaggleスタートブック](https://www.amazon.co.jp/dp/4065190061)
2. [kaggleで勝つデータ分析の技術](https://www.amazon.co.jp/dp/4297108437)
3. [4GM本:Approaching (Almost) Any Machine Learning Problem](https://www.amazon.co.jp/dp/8269211508)

本書籍は「kaggleで勝つデータ分析の技術」と内容が重複している部分も多いが、ユニークな点として下記があると感じた。
- 特徴量選択の章がある
- CVやNLPに特化した章がある
- 実運用を見据えたserving方法にも触れている


## 所感
- 知らない関数（e.g. numpy.ptpやsklearnのimpute）を知ることができた。
- 自分の知識の隙間を埋めてくれるような内容であった。
- 当たり前だけど、基本を忠実にこなすのが大切なのだな、と実感した。


## 名言っぽいものを残しておく
- If you want to do feature engineering, split your data first. If you're going to build models, split your data first.
- Please note that it’s usually better to create less and important features than to create hundreds of features in the first place.