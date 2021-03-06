---
title: "#7 mlflow使ってみた"
date: 2020-06-27T00:00:00+09:00
draft: false
---

Podcastは[こちら](https://anchor.fm/geek-engineer-future/)から🎵

👉質問コメントは[質問箱](https://peing.net/ja/04affd1e18a05d/message) or [Twitter](https://twitter.com/)のハッシュタグ[`#geek_engineer`](https://twitter.com/search?q=%23geek_engineer)にてお待ちしております📮

## Summary

現在参加中のコンペで、mlflowを使ってみたので、その所感などをお話しました！

## mlflowとは？
- URL: https://mlflow.org/
- 機械学習のプロジェクトを行う上で複雑になりがちな実行環境、モデル、パラメータ、評価結果、その他もろもろの実験管理を行ってくれるプラットフォーム（ライブラリ）
- 以下の3種類がある
    - mlflow Tracking
    - mlflow Projects
    - mlflow Models

### mlflow Tracking
- モデル作成時の実験管理をサポートしてくれる機能

### mlflow Projects
- 作成した機械学習モデル作成コードを誰でも利用できるようにパッケージングする機能

### mlflow Models
- mlflow Trackingで保存したモデルを簡単にデプロイできる機能

## 使ってみた所感
- 実験（run）ごとに各fold毎のスコアだったり、ハイパーパラメーターを管理することができて便利
- しかもそれをブラウザ上で表示して見れるので、どのrunで、どのfoldのスコアがいくつで、それのLBのスコアはいくつだった、みたいなことも一覧表示できる
- runに紐づける形で、Note（フリースタイルのmemo）だったり、Artifact（画像ファイルやtxtファイルなど）を紐づけることも可能
- 導入コストは比較的少なく、普段loggingしている箇所をmlflowに置き換えるだけでOK
