---
title: "#3 自然言語を簡単に可視化・分析できるライブラリ「nlplot」を公開した話"
date: 2020-05-24T00:00:00+09:00
draft: false
---

Podcastは[こちら](https://anchor.fm/geek-engineer-future/episodes/3-nlplot-eefovh)から🎵

👉質問コメントは[質問箱](https://peing.net/ja/04affd1e18a05d/message) or [Twitter](https://twitter.com/)のハッシュタグ[`#geek_engineer`](https://twitter.com/search?q=%23geek_engineer)にてお待ちしております📮

## Summary

自然言語をよしなに可視化・分析できるライブラリ「nlplot」を公開したよ、という話をしました。

## nlplotについて

pipでインストールできます。
```
pip install nlplot
```

下記グラフが簡単に描画できるので、データの概要・傾向を掴みたいときに便利かと思います。

1. [N-gram bar chart](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_uni-gram.html)
2. [N-gram tree Map](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_Tree%20of%20Most%20Common%20Words.html)
3. [Histogram of the word count](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_number%20of%20words%20distribution.html)
4. [wordcloud](https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/wordcloud.png)
5. [co-occurrence networks](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_Co-occurrence%20network.html)
6. [sunburst chart](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_sunburst%20chart.html)
7. [pyLDAvis](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_pyldavis.html)

具体的な使用方法については
- [ブログ](https://www.takapy.work/entry/2020/05/17/192947)
- [kaggleのカーネル](https://www.kaggle.com/takanobu0210/twitter-sentiment-eda-using-nlplot)
- [Githubのサンプルコード](https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/introduction_nlplot_twitter.ipynb)

を参考にしていただければと思います。


## PyPIへの登録は大変？

- 初回は分からないことだらけだったので、結構大変だった。~~しょうもないところで数時間溶かしました~~
    - 数時間溶かした備忘は[ここ](https://www.takapy.work/entry/2020/05/10/155856)にまとめてあります。
- 主に[Python実践入門](https://gihyo.jp/book/2020/978-4-297-11111-3)にかかれている手順を参考に登録した。


## PRお待ちしております〜
あまり綺麗でない状態のコードを公開してしまったので羞恥の念が全身にみなぎっていますが（こんなに伸びると思っていなかった...）今後もアップデートしていきたいと思っていますので、PRなどお待ちしております！

そしてすでにISSUEやPRを送ってくださったみなさま、ありがとうございます！

https://github.com/takapy0210/nlplot