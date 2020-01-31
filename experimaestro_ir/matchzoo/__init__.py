from experimaestro import config, task, argument
from experimaestro_ir import NS
import matchzoo as mz

@argument("model")
@task(NS.mz.learnedmodel)
def MatchZooLearn():
    pass

@task(NS.matchzoo.models.matchpyramid)
def DRMM(MatchZooLearn): 
    def execute(self):
        # From https://github.com/NTMC-Community/MatchZoo-py/blob/master/tutorials/ranking/drmm.ipynb       
        ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=10))
        ranking_task.metrics = [
            mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
            mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
            mz.metrics.MeanAveragePrecision()
        ]
        preprocessor = mz.models.DRMM.get_default_preprocessor()

        ds_glove = prepare_dataset("edu.stanford.glove.6b.300")
        glove_embedding = mz.embedding.load_from_file(ds_glove.path, mode='glove')

        term_index = preprocessor.context['vocab_unit'].state['term_index']
        embedding_matrix = glove_embedding.build_matrix(term_index)
        l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
        embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]

        histgram_callback = mz.dataloader.callbacks.Histogram(
            embedding_matrix, bin_size=30, hist_mode='LCH'
        )

        trainset = mz.dataloader.Dataset(
            data_pack=train_pack_processed,
            mode='pair',
            num_dup=5,
            num_neg=10,
            callbacks=[histgram_callback]
        )
        testset = mz.dataloader.Dataset(
            data_pack=test_pack_processed,
            callbacks=[histgram_callback]
        )

        padding_callback = mz.models.DRMM.get_default_padding_callback()

        trainloader = mz.dataloader.DataLoader(
            device='cpu',
            dataset=trainset,
            batch_size=20,
            stage='train',
            resample=True,
            callback=padding_callback
        )
        testloader = mz.dataloader.DataLoader(
            dataset=testset,
            batch_size=20,
            stage='dev',
            callback=padding_callback
        )

        model = mz.models.DRMM()

        model.params['task'] = ranking_task
        model.params['mask_value'] = 0
        model.params['embedding'] = embedding_matrix
        model.params['hist_bin_size'] = 30
        model.params['mlp_num_layers'] = 1
        model.params['mlp_num_units'] = 10
        model.params['mlp_num_fan_out'] = 1
        model.params['mlp_activation_func'] = 'tanh'

        model.build()

        print(model)
        print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

        optimizer = torch.optim.Adadelta(model.parameters())

        trainer = mz.trainers.Trainer(
            device='cpu',
            model=model,
            optimizer=optimizer,
            trainloader=trainloader,
            validloader=testloader,
            validate_interval=None,
            epochs=10
        )

        trainer.run()