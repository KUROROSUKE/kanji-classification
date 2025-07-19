対象：日本語の手書き常用漢字  
データベース：ETL9B（http://etlcdb.db.aist.go.jp/）  
  
・URLからデータをダウンロードしてETL9Bフォルダに保存　
・ETL9B.pyを実行するとETL9Bからデータを読み込んでcharsに各漢字ごとのフォルダとして保存される  
・NN.pyを実行してモデルを学習＆保存（今回はMODEL/NN.pthに保存済み）  
・saveChar.htmlをLive Serverか[サイト](https://kurorosuke.github.io/kanji-classification/saveChar.html)で漢字を書いて保存  
・保存した画像のパスを指定してpredict.pyを実行して予測  
