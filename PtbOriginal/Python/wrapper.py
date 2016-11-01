import subprocess

def basic_params():
    vocab_size="10000" 
    optimizer="GradDesc" 
    keep_prob="0.5" 
    max_grad_norm="5" 
    num_layers="2" 
    batch_size="20" 
    init_scale="0.05" 
    max_epoch="6" 
    num_steps="35" 
    max_max_epoch="39" 
    lr_decay="0.8" 
    loss_function="sequence_loss_by_example" 
    hidden_size="512" 
    learning_rate="1" 
    embedded_size="256" 
    return learning_rate, embedded_size, vocab_size, optimizer, keep_prob, max_grad_norm, num_layers, batch_size, init_scale, max_epoch, num_steps, max_max_epoch, lr_decay, loss_function, hidden_size
    
learning_rate, embedded_size, vocab_size, optimizer, keep_prob, max_grad_norm, num_layers, batch_size, init_scale, max_epoch, num_steps, max_max_epoch, lr_decay, loss_function, hidden_size = basic_params()

test_name = 'lelijk'
num_run = '12'
subprocess.call(' python ptboriginal.py --vocab_size="' + vocab_size +'" --optimizer="' + optimizer + '" --keep_prob="'+keep_prob+'" --max_grad_norm="'+max_grad_norm +'" --num_layers="'+num_layers+'" --batch_size="'+batch_size+'" --init_scale="' + init_scale+ '" max_epoch="' + max_epoch +'" --num_steps="'+num_steps+'" --max_max_epoch="'+max_max_epoch+'" --lr_decay="'+lr_decay+'" --loss_function="'+loss_function + '" --hidden_size="'+hidden_size +'" --learning_rate="' + learning_rate + '" --embedded_size="' + embedded_size + '" --num_run="' +num_run + '" --test_name="' + test_name +'"', shell=True)
