scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        loss = criterion(model(xb), yb)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
