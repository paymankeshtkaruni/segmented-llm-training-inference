# Cover letter — JSA VSI "Security and Efficiency for LLM-Based Edge Intelligence"

Dear Prof. Meikang Qiu and Prof. Wenqi Wei,

Please consider our manuscript, "Segmented Execution of Large Language
Models: Sequential Training and Inference on Resource-Constrained Devices,"
for the Journal of Systems Architecture special issue on Security and
Efficiency for LLM-Based Edge Intelligence.

The special issue calls for new optimization techniques that improve the
efficiency and scalability of LLM-based edge intelligence under
resource-constrained conditions. Our manuscript addresses exactly this gap,
for the hardest case: not only inference but full-parameter training of
LLMs on devices whose memory is far below the model's conventional
footprint. The model is partitioned along four axes (attention head groups,
feed-forward hidden units, embedding width, and vocabulary slices) and
executed one segment at a time, so that at most one small segment is ever
resident on the constrained device. A small set of execution dials defines
twelve training and four inference operating modes, each measured on GPU
and CPU; every mode is verified — to machine precision, against a control
that measures the training framework's own run-to-run noise — to compute
the same model.

Measured capability results include: a 0.84-billion-parameter decoder
trained in 1.08 GB of device memory (23x below full-model training and 20x
below the DeepSpeed ZeRO-Offload floor measured on the same node); a
6.9-billion-parameter model — whose full-model training aborts even on a
94 GB accelerator — trained on a CPU-only node in 9.3 GB of RAM, or on a
single 40 GB accelerator; and a serving mode that answers queries in
401 MB of total memory without a GPU or a training framework. Analytic
memory and time laws are validated by a pre-registered prediction on a
held-out configuration.

The manuscript also speaks to the security half of the special issue's
scope: because both training and inference run entirely on the device that
owns the data, no raw records, gradients, or activations ever cross a
network boundary, and the mechanism is the enabling substrate for federated
learning over resource-constrained participants, which we outline as future
work.

All results are reproducible: every reported number is produced by a
committed script and batch-job specification in the accompanying
repository, and the cost-model validation was pre-registered by committing
its prediction before the measurement ran.

This manuscript is original, is not under consideration elsewhere, and all
authors have approved the submission. We have no competing interests to
declare. This work was supported by the Federal Ministry of Education and
Research (BMBF), Germany, under the AI service center KISSKI (grant nos.
01IS22093A and 01IS22093B).

Thank you for your consideration.

On behalf of all authors,

Sadegh Keshtkar (corresponding author)
Institute of Computer Science, University of Göttingen
keshtkar.sadegh@gmail.com
