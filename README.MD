# Re-Implementation Large-Language Models are Zero-Shot Time Forecasters.

Welcome to the evaluation report for the re-implementation of LLMTime [1]. This repository delves into the
analysis and assessment conducted on the recreated open-source implementation. The
primary objective of this endeavor is to offer a transparent and detailed walkthrough of the pre-processing,
hyper-parameter search, and prediction processes from the original LLM Time work, by following the details of the
open-access version.

This implementation attempts to use as much default functions, trying to minimize 'hacky' solutions.
In the spirit of robustness, a series of tests were developed during debugging to validate various components of our
re-implementation, which can be found in [`test`](./test).


This contribution to the open-source community is designed not only to be a faithful recreation but also to serve as an
extensible template. Our intention is to empower others to explore the concept of LLMTime on their local machines,
fostering a collaborative environment for further innovation. 

Feel free to open a pull request or issue if you find something interesting/broken/have a question!


## Running the code

To run the experiments, you can install the package using `pip` package manager.

To run all experiments, you can run the [`run.sh`](./run.sh) file, which will automatically run all the experiments.
Make sure to set hour `HUGGING_FACE_TOKEN` in your environment as `HF_TOKEN=hf_.....`, or set it in your `.pyenv`.

To analyze the results, move all the generated data to [`results`](results), and run the [`plotter.ipynb`](./plotter.ipynb)
notebook to generate figures.

For generated plots (on my machine), you can see the pdfs in the [`plots`](./plots) folder!


## Hardware requirements

This code can run on an RTX3090, as we use half-precision model. GPT2 can be run on considerably less
powerful hardware.

You can decrease the parallelism of the generation in the code if your processes get OOMKilled, or reduce
the context you provide in the train/test split.