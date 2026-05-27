from model_architecture import Model, InitialCNN



def run_model(model):
    trainer = Model(model)
    results = trainer.evaluate()
    trainer.plot_learning_curves(*results[:4])
    trainer.plot_gradients(results[4])
    trainer.show_samples()


print('Model before improvement:')
run_model(InitialCNN())

print('////////////////////////////////////////////////////////////////////////////////////////////////////')

print('Model after improvement:')
run_model(InitialCNN(True))