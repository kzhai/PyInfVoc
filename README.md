PyInfVoc
==========

PyInfVoc is an Online Latent Dirichlet Allocation with Infinite Vocabulary topic modeling package based on Variational Bayesian learning approach under online settings, developed by the Cloud Computing Research Team in [University of Maryland, College Park] (http://www.umd.edu). You may find more details about this project on our papaer [Online Latent Dirichlet Allocation with Infinite Vocabulary] (http://kzhai.github.io/paper/2013_icml.pdf) appeared in ICML 2013.

Please download the latest version from our [GitHub repository](https://github.com/kzhai/PyInfVoc).

Please send any bugs of problems to Ke Zhai (kzhai@umd.edu).

Install and Build
----------

This package depends on many external python libraries, such as numpy, scipy and nltk. After downloading the source code packages, unzip the datasets to the 'input' directory. The package includes a few fundamental datasets --- ap, de-news and 20-newsgroup datasets.

Launch and Execute
----------

Assume the PyInfVoc package is downloaded under directory ```$PROJECT_SPACE/src/```, i.e., 

	$PROJECT_SPACE/src/PyInfVoc

To prepare the example dataset,

	tar zxvf de-news.tar.gz

To launch PyInfVoc, first redirect to the directory of PyInfVoc source code,

	cd $PROJECT_SPACE/src/PyInfVoc

and run the following command on example dataset,

	python -m launch_train --input_directory=./de-news/ --output_directory=./ --truncation_level=4000 --number_of_topics=10 --number_of_documents=9800 --vocab_prune_interval=10 --batch_size=98 --alpha_beta=100
	
The generic argument to run PyLDA is

	python -m launch_train --input_directory=$INPUT_DIRECTORY/$CORPUS_NAME --output_directory=$OUTPUT_DIRECTORY --number_of_topics=$NUMBER_OF_TOPICS --number_of_documents=$NUMBER_OF_DOCUMENTS --batch_size=$BATCH_SIZE

You should be able to find the output at directory ```$OUTPUT_DIRECTORY/$CORPUS_NAME```.

Under any cirsumstances, you may also get help information and usage hints by running the following command

	python -m launch_train --help
