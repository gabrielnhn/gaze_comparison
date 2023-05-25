        for epoch in range(num_epochs):
            avg_error_train = 0.0
            total_train = 0
            sum_loss_pitch_gaze = 0
            sum_loss_yaw_gaze = 0
            iter_gaze = 0
            
            model.train()
            for i, (images_gaze, labels_gaze, cont_labels_gaze,name) in enumerate(train_loader_gaze):

                # total_train += cont_labels_gaze.size(0)
                total_train += 2 * cont_labels_gaze.size(0)
                images_gaze = Variable(images_gaze).cuda(gpu)
                
                # Binned labels
                label_yaw_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                label_pitch_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)
                # Continuous labels
                label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                #Mirror
                mirror_image = images_gaze.detach().clone()
                for i in range(len(mirror_image)):
                    mirror_image[i] = torchvision.transforms.functional.hflip(mirror_image[i])

                mirror_yaw_bin = [(model.num_bins -1 - binned_yaw) for binned_yaw in label_yaw_gaze]
                mirror_pitch_bin = [int(binned_pitch) for binned_pitch in label_pitch_gaze]
                mirror_pitch_cont = [pitch for pitch in label_pitch_cont_gaze]
                mirror_yaw_cont = [-yaw for yaw in label_yaw_cont_gaze]

                # mirror_image = Variable(torch.Tensor(mirror_image)).cuda(gpu)
                mirror_image = Variable(mirror_image).cuda(gpu)
                mirror_yaw_bin = Variable(torch.tensor(mirror_yaw_bin)).cuda(gpu)
                mirror_pitch_bin = Variable(torch.tensor(mirror_pitch_bin)).cuda(gpu)
                mirror_pitch_cont = Variable(torch.Tensor(mirror_pitch_cont)).cuda(gpu)
                mirror_yaw_cont = Variable(torch.Tensor(mirror_yaw_cont)).cuda(gpu)

                ##CALCULATE ORIGINAL

                yaw_predicted, pitch_predicted = model(images_gaze)

                # Cross entropy loss
                loss_pitch_gaze = criterion(pitch_predicted, label_pitch_gaze)
                loss_yaw_gaze = criterion(yaw_predicted, label_yaw_gaze)

                with torch.no_grad():
                    pitch_predicted_cpu = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * binwidth - 180
                    yaw_predicted_cpu = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * binwidth - 180
                    label_pitch_cpu = cont_labels_gaze[:,1].float()*np.pi/180
                    label_pitch_cpu = label_pitch_cpu.cpu()
                    label_yaw_cpu = cont_labels_gaze[:,0].float()*np.pi/180
                    label_yaw_cpu = label_yaw_cpu.cpu()

                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * binwidth - 180
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * binwidth - 180

                loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont_gaze)
                loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont_gaze)

                # Total loss
                loss_pitch_gaze += alpha * loss_reg_pitch
                loss_yaw_gaze += alpha * loss_reg_yaw

                loss_seq = [loss_yaw_gaze, loss_pitch_gaze]
                grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
                optimizer_gaze.zero_grad(set_to_none=True)
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer_gaze.step()

                with torch.no_grad():
                    pitch_predicted_cpu = pitch_predicted_cpu*np.pi/180
                    yaw_predicted_cpu = yaw_predicted_cpu*np.pi/180 

                    for p,y,pl,yl in zip(pitch_predicted_cpu,yaw_predicted_cpu,label_pitch_cpu,label_yaw_cpu):
                        avg_error_train += angular(gazeto3d([p,y]), gazeto3d([pl,yl]))
 

                ####### CALCULATE MIRROR
                yaw_predicted, pitch_predicted = model(mirror_image)

                # Cross entropy loss
                loss_pitch_gaze = criterion(pitch_predicted, mirror_pitch_bin)
                loss_yaw_gaze = criterion(yaw_predicted, mirror_yaw_bin)

                with torch.no_grad():
                    pitch_predicted_cpu = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * binwidth - 180
                    yaw_predicted_cpu = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * binwidth - 180

                    # label_pitch_cpu = cont_labels_gaze[:,1].float()*np.pi/180
                    label_pitch_cpu = mirror_pitch_cont.float()*np.pi/180
                    label_pitch_cpu = label_pitch_cpu.cpu()
                    # label_yaw_cpu = cont_labels_gaze[:,0].float()*np.pi/180
                    label_yaw_cpu = mirror_yaw_cont.float()*np.pi/180
                    label_yaw_cpu = label_yaw_cpu.cpu()

                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * binwidth - 180
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * binwidth - 180

                loss_reg_pitch = reg_criterion(pitch_predicted, mirror_pitch_cont)
                loss_reg_yaw = reg_criterion(yaw_predicted, mirror_yaw_cont)

                # Total loss
                loss_pitch_gaze += alpha * loss_reg_pitch
                loss_yaw_gaze += alpha * loss_reg_yaw

                loss_seq = [loss_yaw_gaze, loss_pitch_gaze]
                grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
                optimizer_gaze.zero_grad(set_to_none=True)
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer_gaze.step()

                with torch.no_grad():
                    pitch_predicted_cpu = pitch_predicted_cpu*np.pi/180
                    yaw_predicted_cpu = yaw_predicted_cpu*np.pi/180 

                    for p,y,pl,yl in zip(pitch_predicted_cpu,yaw_predicted_cpu,label_pitch_cpu,label_yaw_cpu):
                        avg_error_train += angular(gazeto3d([p,y]), gazeto3d([pl,yl]))
"