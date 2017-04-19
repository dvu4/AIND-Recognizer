LAST UPDATE: 10.02.2011
______________________________________________________________________


The RWTH-BOSTON-104 Database:
-----------------------------

The National Center for sign language and Gesture Resources of the
Boston University published a database of ASL sentences. Although this
database has not been produced primarily for image processing research,
it consists of 201 annotated video streams of ASL sentences.

The signing is captured simultaneously by four standard stationary
cameras where three of them are black/white and one is a color camera.
Two black/white cameras, placed towards the signer's face, form a stereo
pair and another camera is installed on the side of the signer. The
color camera is placed between the stereo camera pair and is zoomed
to capture only the face of the signer. The movies published on the
Internet are at 30 frames per second, the size of the videos is 366x312
pixels, and the size of the frames without any additional recording
information is 312x242 pixels (i.e. the ROI). We use the published video
streams at the same frame rate but we use only the upper center part
of size 195x165 pixels, with top-left corner at (x,y)=(70,10), because
parts of the bottom of the frames show some information about the frame
and the left and right border of the frames are unused.

To create the RWTH-BOSTON-104 database for ASL sentence recognition, we
separated the video streams into a training and test set. The training
set consists of 161 sign language sentences and the test set includes
the 40 remaining sign language sentences. Furthermore, the training set
was splited again into a smaller training set with 131 sequences and a
development set with 30 sequences, in order to tune the parameters of
the system.

In [1], a 17.9% word-error-rate (WER) was achieved on this database. The
same experiment protocol as in [1] was used in [3], where a Model-based
Joint Tracking and Recognition approach could further decrease the WER
down to 11.24% WER, which is also the currently best known WER on this
database.

PS: as frame exact seeking in videos is not always trivial in video
processing, and the number of total extracted frames per video often
depends on the used codecs (e.g. libmpeg2 vs. libffmpeg), we suggest
you to use the already extracted png-segments, which were used in our
experiments too. If you want to work with videos, use the corresponding
corpus/mpg.* corpora files.

______________________________________________________________________

In the corpus/ folder, the are several corpus files which can be used
to reproduce experiments:

File: devel-test.corpus                         7 KB    06/06/2007      12:00:00 AM
File: devel-train.corpus                        27 KB   06/06/2007      12:00:00 AM
File: speaker.description                       1 KB    06/06/2007      12:00:00 AM
File: test.sentences.corpus                     9 KB    06/06/2007      12:00:00 AM
File: train.sentences.pronunciations.corpus     33 KB   06/06/2007      12:00:00 AM


We presented two experiment setups in our publications:

1. Interspeech 2007 Experiment [1]:
-------------------------------------------------------------
training corpus:        corpus/train.sentences.pronunciations.corpus
training lexicon:       lexicon/train_0.12.lexicon

testing corpus:         corpus/test.sentences.corpus
testing lexicon:        lexicon/test_0.12.lexicon
3-gram LM:              lm/ukn.3.lm.gz


Experiment Protocol Overview:
-----------------------------
The system was trained using the full training set (161 seq):
	corpus/train.sentences.pronunciations.corpus
with the training lexicon:
        lexicon/train_0.12.lexicon
and evaluated using the unseen test set (40 seq):
	corpus/test.sentences.corpus
with a testing lexicon:
	lexicon/test_0.12.lexicon
and with a corresponding 3-gram language model in:
	lm/ukn.3.lm.gz




2. BMVC 2006 Experiment [2]:
-------------------------------------------------------------

1.1 first pass: tune parameters on hold out set
---------------
training:               corpus/devel-train.corpus
training lexicon:       lexicon/devel-train_0.12.lexicon

tuning corpus:          corpus/devel-test.corpus
tuning lexicon:         lexicon/devel-test_0.12.lexicon
3-gram LM:              lm/devel-lm-M3.sri.lm.gz

1.2 second pass:  evaluation with optimized parameters and retraining using full training corpus
----------------
retraining:             corpus/train.sentences.pronunciations.corpus
training lexicon:       lexicon/train_0.12.lexicon

testing:                corpus/test.sentences.corpus
testing lexicon:        lexicon/test_0.12.lexicon
3-gram LM:              lm/ukn.3.lm.gz


Experiment Protocol Overview:
-----------------------------

The system was trained on (131 seq):    
        corpus/devel-train.corpus
with a training lexicon:	        
        lexicon/devel-train_0.12.lexicon
and tuned using the development set (30 seq):
	corpus/devel-test.corpus
with a testing lexicon:
	lexicon/devel-test_0.12.lexicon
and with the corresponding 3-gram language model in:
	lm/devel-lm-M3.sri.lm.gz


Then after tuning the parameters, the system was retrained using the
full training set (161 seq):
	corpus/train.sentences.pronunciations.corpus
with the training lexicon:
        lexicon/train_0.12.lexicon
and evaluated using the best parameters on the unseen test set (40 seq):
	corpus/test.sentences.corpus
with a testing lexicon:
	lexicon/test_0.12.lexicon
and with a corresponding 3-gram language model in:
	lm/ukn.3.lm.gz


______________________________________________________________________
RWTH-BOSTON-Hands and further Groundtruth Annotations for Tracking Experiments:

The positions of the left hand, the right hand, and the nose have been
annotated for all 15k frames of the RWTH-BOSTON-104 database. The older 
RWTH-BOSTON-Hands database is a subset of the full annotation file.

Full annotation file:
handpositions/boston104-hands.handpositions.rybach-forster-dreuw-2009-09-25.full.xml

To evaluate your tracking results on the old RWTH-BOSTON-Hands database, 
you should have a look at [4] and the description of the database. The 
corresponding corpus file, which is a subset of the RWTH-BOSTON-104
corpus, for these experiments is:
handpositions/boston104-hands.corpus

To evaluate your tracking results on the full RWTH-BOSTON-104 database, 
you should have a look at [5].

______________________________________________________________________


References:
-----------

You should cite at least one of the following works if you publish your
results achieved on this database:

[1] Philippe Dreuw, David Rybach, Thomas Deselaers, Morteza Zahedi,
Hermann Ney. Speech Recognition Techniques for a Sign Language
Recognition System. In Interspeech 2007, pages 2513-2516, Antwerp,
Belgium, August 2007.

@InProceedings { dreuw:interspeech2007,
  author= {Dreuw, P. and Rybach, D. and Deselaers, T. and Zahedi, M. and Ney, H.},
  title= {Speech Recognition Techniques for a Sign Language Recognition System},
  booktitle= {Interspeech 2007},
  year= 2007,
  pages= {2513-2516},
  address= {Antwerp, Belgium},
  month= aug,
  note= {ISCA best student paper award Interspeech 2007},
  booktitlelink= {http://www.interspeech2007.org},
  pdf = {http://www-i6.informatik.rwth-aachen.de/publications/downloader.php?id=154&row=pdf},
  aux = {http://www-i6.informatik.rwth-aachen.de/publications/downloader.php?id=154&row=aux},
  url = {http://www.isca-speech.org/awards.html#student}
}


[2] Morteza Zahedi, Philippe Dreuw, David Rybach, Thomas Deselaers,
Hermann Ney. Continuous Sign Language Recognition - Approaches from
Speech Recognition and Available Data Resources. In Second Workshop
on the Representation and Processing of Sign Languages: Lexicographic
Matters and Didactic Scenarios, pages 21-24, Genoa, Italy, May 2006.


@InProceedings { zahedi06bmvc,
  author= {Zahedi, M. and Dreuw, P. and Rybach, D. and Deselaers, T. and Ney, H.},
  title= {Using Geometric Features to Improve Continuous Appearance-based Sign Language Recognition},
  booktitle= {17th British Maschine Vision Conference},
  year= 2006,
  pages= {1019-1028},
  address= {Edinburgh, UK},
  month= sep,
  volume= {3},
  booktitlelink= {http://www.macs.hw.ac.uk/bmvc2006/},
  pdf = {http://www-i6.informatik.rwth-aachen.de/publications/downloader.php?id=77&row=pdf}
}


[3] P. Dreuw, J. Forster, T. Deselaers, and H. Ney. Efficient
Approximations to Model-based Joint Tracking and Recognition of
Continuous Sign Language. In IEEE International Conference Automatic
Face and Gesture Recognition (FG), Amsterdam, The Netherlands, September
2008.

@InProceedings { dreuw:EfficientApproxJointTrackingRecognition:2008,
author= {Dreuw, Philippe and Forster, Jens and Deselaers, Thomas and Ney, Hermann},
title= {Efficient Approximations to Model-based Joint Tracking and Recognition of Continuous Sign Language},
booktitle= {IEEE International Conference Automatic Face and Gesture Recognition},
year= 2008,
address= {Amsterdam, The Netherlands},
month= sep,
booktitlelink= {http://www.fg2008.nl/},
pdf = {http://www-i6.informatik.rwth-aachen.de/publications/downloader.php?id=560&row=pdf},
aux = {http://www-i6.informatik.rwth-aachen.de/publications/downloader.php?id=560&row=aux}
}


[4] D. Rybach. Appearance-Based Features for Automatic Continuous Sign
Language Recognition. Master Thesis, Aachen, Germany, June 2006.

@Mastersthesis { rybach06:diploma,
author= {Rybach, David},
title= {Appearance-Based Features for Automatic Continuous Sign Language Recognition},
school= {Human Language Technology and Pattern Recognition Group, RWTH Aachen University},
year= 2006,
address= {Aachen, Germany},
month= jun,
pdf = {http://www-i6.informatik.rwth-aachen.de/publications/downloader.php?id=71&row=pdf}
}


[5] P. Dreuw, J. Forster, and H. Ney. Tracking Benchmark Databases for
Video-Based Sign Language Recognition. In ECCV International Workshop on
Sign, Gesture, and Activity (SGA), Crete, Greece, September 2010.

@InProceedings { dreuw:TrackingBenchmarks:sga2010,
author= {Dreuw, Philippe and Forster, Jens and Ney, Hermann},
title= {Tracking Benchmark Databases for Video-Based Sign Language Recognition},
booktitle= {ECCV International Workshop on Sign, Gesture, and Activity},
year= 2010,
address= {Crete, Greece},
month= sep,
booktitlelink= {http://personal.ee.surrey.ac.uk/Personal/R.Bowden/SGA2010/},
pdf = {http://www-i6.informatik.rwth-aachen.de/publications/downloader.php?id=673&row=pdf}
}

______________________________________________________________________

Contact:
--------

If you have any questions concerning the setup of this database, please
contact 

	Philippe Dreuw <dreuw@cs.rwth-aachen.de>.

The database is freely available at

	http://www-i6.informatik.rwth-aachen.de/~dreuw/database.php
______________________________________________________________________
