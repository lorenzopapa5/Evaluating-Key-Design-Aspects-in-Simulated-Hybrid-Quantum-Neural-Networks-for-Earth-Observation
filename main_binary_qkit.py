import torch, os, argparse
from utils import set_seed
from DatasetHandler import DatasetHandler, data_loader
# from training_functions import stardard_train, init_and_qtrain
from training_functions import stardard_train
from evaluate import evaluate_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_data(dataset_path, classes, batch_size, img_shape, n_work=4, train_val_split=0.2, if_print=False):
    datasetHandler = DatasetHandler(dataset_path)
    if if_print:
        datasetHandler.print_classes()

    classes_name = []
    classes_name.append(datasetHandler.classes[classes[0]].split('/')[-1])
    classes_name.append(datasetHandler.classes[classes[1]].split('/')[-1])

    imgs_path, imgs_label = datasetHandler.load_paths_labels(classes = [datasetHandler.classes[classes[0]], datasetHandler.classes[classes[1]]])
    if if_print:
        print('Dataset images: ', len(imgs_path))
        print('Dataset labels: ', len(imgs_label))
        print('Dataset sample -> image path: ', imgs_path[0], ' image label', imgs_label[0])

    train_images, train_labels, val_images, val_labels = datasetHandler.train_validation_split(imgs_path, imgs_label, split_factor=train_val_split)
    if if_print:
        print('Training images: ',  train_images.shape)
        print('Training labels: ',  train_labels.shape)
        print('Validatiom images: ',  val_images.shape)
        print('Validation labels: ',  val_labels.shape)

    train_loader = data_loader(train_images, train_labels, batch_size=batch_size, img_shape=img_shape, shuffle=True, num_workers=n_work)
    val_loader = data_loader(val_images, val_labels, batch_size=1, img_shape=img_shape, shuffle=False, num_workers=n_work)
    
    return train_loader, val_loader


if __name__ == '__main__':

    # Setup in cmd line
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True) 
    parser.add_argument('--c1', type=int, required=True)
    parser.add_argument('--c2', type=int, required=True)
    parser.add_argument('--ep', type=int, default=20)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1699806)
    parser.add_argument('--trial', type=int, default=0)

    parser.add_argument('--init_model_and_full_retrain_with_quantum', action='store_true')
    parser.add_argument('--pretrained_model_and_only_quantum_train', action='store_true')
    
    args = parser.parse_args()

    # Variables
    dataset_path = '/work/dataset/EuroSAT/'
    classes = [args.c1, args.c2]
    img_shape = (3, 64, 64)
    batch_size = args.bs
    epochs = args.ep
    seed = args.seed
    model_name = args.model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed=seed)
    train_loader, val_loader = load_data(dataset_path, classes, batch_size=batch_size, img_shape=img_shape, if_print=True)

    save_path = '/work/save_models/' + args.model_name + '/' + 'c' + str(args.c1) + '_c' + str(args.c2) + '_seed:' + str(args.seed) + '_ep:' + str(args.ep) + '_bs:' + str(args.bs) + '_trial' + str(args.trial) + '/' 
    os.makedirs(save_path, exist_ok=True)
    final_model = stardard_train(model_name, train_loader, val_loader, epochs, batch_size, img_shape, save_path, device)

    """ OLD CODE NEEDED FOR PARTICUALR TRAINING

    if "q" in model_name or args.init_model_and_full_retrain_with_quantum or args.pretrained_model_and_only_quantum_train:
        device = 'cpu'

    if args.init_model_and_full_retrain_with_quantum:
        device = 'cpu'
        epochs = 10
        save_path = '/work/save_models/' + args.model_name + '_init_and_full_qtrain/' + 'c' + str(args.c1) + '_c' + str(args.c2) + '_seed:' + str(args.seed) + '_ep:' + str(args.ep) + '_bs:' + str(args.bs) + '_trial' + str(args.trial) + '/'
        pretrained_model_path = '/work/save_models/' + args.model_name + '/' + 'c' + str(args.c1) + '_c' + str(args.c2) + '_seed:' + str(args.seed) + '_ep:' + str(args.ep) + '_bs:' + str(args.bs) + '_trial' + str(args.trial) + '/best_model.pth' 
        os.makedirs(save_path, exist_ok=True)
        final_model = init_and_qtrain(model_name, train_loader, val_loader, epochs, batch_size, img_shape, save_path, pretrained_model_path, False, device)

    elif args.pretrained_model_and_only_quantum_train:
        device = 'cpu'
        epochs = 5
        save_path = '/work/save_models/' + args.model_name + '_init_and_only_qtrain/' + 'c' + str(args.c1) + '_c' + str(args.c2) + '_seed:' + str(args.seed) + '_ep:' + str(args.ep) + '_bs:' + str(args.bs) + '_trial' + str(args.trial) + '/' 
        pretrained_model_path = '/work/save_models/' + args.model_name + '/' + 'c' + str(args.c1) + '_c' + str(args.c2) + '_seed:' + str(args.seed) + '_ep:' + str(args.ep) + '_bs:' + str(args.bs) + '_trial' + str(args.trial) + '/best_model.pth' 
        os.makedirs(save_path, exist_ok=True)
        final_model = init_and_qtrain(model_name, train_loader, val_loader, epochs, batch_size, img_shape, save_path, pretrained_model_path, True, device)

    else:   
        save_path = '/work/save_models/' + args.model_name + '/' + 'c' + str(args.c1) + '_c' + str(args.c2) + '_seed:' + str(args.seed) + '_ep:' + str(args.ep) + '_bs:' + str(args.bs) + '_trial' + str(args.trial) + '/' 
        os.makedirs(save_path, exist_ok=True)
        final_model = stardard_train(model_name, train_loader, val_loader, epochs, batch_size, img_shape, save_path, device)
    """
    
    evaluate_model(final_model, val_loader, device, save_path)

    print('\n ----- Process Completed ----- #\n')
