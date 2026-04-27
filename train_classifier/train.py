import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os
from argparse import ArgumentParser
from tqdm import tqdm
from utils import load_dataset, ResNet50_BinaryClassifier


def train_model(args, model, train_dataloader, val_dataloader, num_epochs, lr, device=None):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # 初始化最佳准确率和最佳模型权重
    best_val_accuracy = 0.0
    best_model_weights = None

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        model.train()
        for inputs, labels in tqdm(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            adv_inputs = inputs.detach()

            optimizer.zero_grad()
            outputs = model(adv_inputs).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        total_loss /= len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss:.4f}, Train Accuracy: {accuracy:.2f}%")

        # Evaluation on the validation set
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_dataloader):
                # labels = labels.type(torch.FloatTensor)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).to(device)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_accuracy = 100 * correct / total
        val_loss /= len(val_dataloader)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_weights = model.state_dict().copy()
            print(f"New best model found! Validation accuracy: {val_accuracy:.2f}%")

    # 保存最终模型
    final_ckpt_save_dir = os.path.join(args.out_dir, f'{args.model_name}_{args.wm_method}.pth')
    torch.save(model.state_dict(), final_ckpt_save_dir)
    print(f"Final model saved to {final_ckpt_save_dir}")

    # 保存最佳模型
    if best_model_weights is not None:
        best_ckpt_save_dir = os.path.join(args.out_dir, f'{args.model_name}_{args.wm_method}_best.pth')
        torch.save(best_model_weights, best_ckpt_save_dir)
        print(f"Best model saved to {best_ckpt_save_dir} with validation accuracy: {best_val_accuracy:.2f}%")


# Main function to run the binary classification task
def main():
    parser = ArgumentParser()
    parser.add_argument("--wm_method", default="StrongMark_denoise_1", type=str,
                        # choices=["dwtDct", "dwtDctSvd", "rivaGan", "SSL", "stegaStamp", "ZoDiac"]
                        )
    parser.add_argument("--model_name", type=str, default="ResNet50")
    parser.add_argument("--wm_dir", type=str, default="StrongMark_results/denoise_steps_1/wm_tmp")
    parser.add_argument("--org_dir", type=str, default="dataset/imagenet_compatible_org")
    parser.add_argument("--out_dir", type=str, default="classifier_ckpt/StrongMark")
    parser.add_argument("--data_cnt", default=100, type=int)
    parser.add_argument("--gpuNum", default=2, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    args = parser.parse_args()

    # args.wm_dir = os.path.join(args.wm_dir, f'{args.wm_method}')
    assert os.path.exists(args.wm_dir)
    print(f'{args.wm_dir}')
    os.makedirs(args.out_dir, exist_ok=True)

    print('==> loading dataset ...')
    train_set, test_set = load_dataset(args)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)  # check
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)  # check

    ## Create the model and train it
    model = ResNet50_BinaryClassifier()

    device = torch.device(f"cuda:{args.gpuNum}" if torch.cuda.is_available() else "cpu")
    print('==> training ...')
    train_model(args, model, train_loader, val_loader, num_epochs=args.epochs, lr=1e-5, device=device)


if __name__ == "__main__":
    main()
