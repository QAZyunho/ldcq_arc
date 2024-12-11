# ldcq_arc

## LDCQ with SOLAR vs Own data

If you want to use SOLAR to train an agent, refer to [SOLAR-Generator](SOLAR-Generator/).  
If you want to use your own dataset, the dataset should have a structure like SOLAR. Or, you can change the dataloader part. 
The dataloader loads data from a folder structured as shown below.
```
├── segment
│   ├── ${task_id}
│   │   ├──${task_id}_${trajectory_id}
│   │   │   │   ├──${task_id}_${trajectory_id}_${segment_id}.json
│   │   │   │   └── ...
```

Our code uses dataloader below.
```
class ARC_Segment_Dataset(Dataset):
    
    def __init__(self, data_path): 
        self.data_path = Path(data_path)

        self.file_list = []

        for path, _, files in os.walk(self.data_path):
            for name in files:
                self.file_list.append(os.path.join(path, name))

        self.num_dataset = len(self.file_list)
        
        if(self.num_dataset == 0):
            raise ValueError("Wrong data path or empty folder. Please check the data path.")
        else:
            print("Number of episodes: {0}".format(self.num_dataset))
        
    def __len__(self):
        return self.num_dataset

    def __getitem__(self, idx):
        trace_file = self.file_list[idx]
            
        with open(trace_file, 'r') as f:
            trace = json.load(f)
        
        state = torch.FloatTensor(trace['grid'])
        selection = torch.LongTensor(trace['selection']).unsqueeze(-1)
        operation = torch.LongTensor(trace['operation']).unsqueeze(-1)
        reward = torch.FloatTensor(trace['reward'])
        terminated = torch.LongTensor(trace['terminated'])
        selection_mask = torch.LongTensor(trace['selection_mask'])
        
        in_grid = torch.FloatTensor(trace['in_grid']).unsqueeze(0)
        out_grid = torch.FloatTensor(trace['out_grid']).unsqueeze(0)
        
        clip = torch.FloatTensor(trace['clip'])
        clip_dim = torch.LongTensor(trace['clip_dim'])
        
        ex_in = torch.FloatTensor(trace['ex_in'])
        ex_out = torch.FloatTensor(trace['ex_out'])
        
        s_T= torch.FloatTensor(trace['next_grid']).unsqueeze(0)
        clip_T = torch.FloatTensor(trace['next_clip']).unsqueeze(0)
        
        return state, s_T, clip, clip_T, selection, operation, reward, terminated, selection_mask, in_grid, out_grid, ex_in, ex_out
```