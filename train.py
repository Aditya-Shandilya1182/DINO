import torch

def train_dino(student, teacher, loss_fn, optimizer, train_loader, num_epochs=20):
    for epoch in range(num_epochs):
        student.train()
        for images, _ in train_loader:
            images = images.cuda()
            student_output = student(images)
            with torch.no_grad():
                teacher_output = teacher(images)

            loss = loss_fn(student_output, teacher_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

def test_model(student, test_loader):
    student.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()

            outputs = student(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
