# Spotify-Like-Predictor
First machine learning project built on sklearn.naive_bayes to estimate whether I would like the song or not.

### Introduction ###     
This is a simple python application which is trying to learn my preferances towards music and then estimate whether I would like the particular song or not. The data set is generated from two csv files:
* Features are fed from songs_i_like.csv
* Labels are fed from labels.csv

songs_i_like.csv is combined of two playlist: 
* [EMOHIPSTER DANCING](https://open.spotify.com/user/yurkoko/playlist/1Ii5NTE1SFSLfbsPzY5KzV) - this is my main Spotify playlist, which I listen to on daily basis
* My roomate's playlist - no link for you because it is terrible.

### Adding more features ###

To build an actual list of features I used an external service [Sort Your Music](http://sortyourmusic.playlistmachinery.com). 

__Note:__ I have no idea if it is reliable, so it is up you if you want to use it.

In the end I've got the following table structure: Id, Title, Artist, Release, BPM, Energy, Dance, Loud, Valence, Length, Acoustic, Pop. 
For labels I used binary code: 
* 0 - Don't like 
* 1 - Like

The final sample contained 272 records.

### Frameworks used ###
* Gaussian Naive Bayes
* pandas
* ateutil.parser

### Accuracy ###

Learning accuracy was 0.75. I think it should be nice for first ever machine learning project.

### Next Steps ###
* Connect to Spotify WEB Api to feed data from Spotify directly instead of using csv-files.
* Enable continuos learning based on "I don't like this song" playlist on Spotify.
* Add search on Spotify for songs to return probability I would like them.
