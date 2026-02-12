import argparse
import os
from pathlib import Path

import pandas as pd

from .preprocessing import mri_preprocess_3d_simple as simple_preproc
from . import get_brainiac_features as extract_features

def extract_filenames(df_col):
    """Replace a column of file paths with just the base file names."""
    df_col = df_col.copy()
    return df_col.apply(os.path.basename).apply(str.strip)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process brain MRI registration and skull stripping.")
    parser.add_argument("--temp_img", type=str, required=True, help="Path to the atlas template image.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input images directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the processed images.")
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/BrainIAC.ckpt',
                      help='Path to the ViT BrainIAC model checkpoint (default: checkpoints/BrainIAC.ckpt)')
    parser.add_argument('--input_csv', type=str, default=None, 
                        help='Path to a CSV which contains data about the input volumes.')
    parser.add_argument('--input_path_col', type=str, default='T1w_path', 
                        help='Column in input_csv which holds the source volume paths')
    parser.add_argument('--output_csv', type=str, required=True,
                      help='Path to save the output features CSV')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for inference (default: 1)')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of workers for data loading (default: 1)')
    args = parser.parse_args()


    """
    !python ./preprocessing/mri_preprocess_3d_simple.py \
    --temp_img ./preprocessing/atlases/temp_head.nii.gz \
    --input_dir ./data/sample/unprocessed \
    --output_dir ./data/sample/processed
    """


    # records DF contains input_path, output_path, id, and status.

    records_df, records_path = simple_preproc.main(temp_img=args.temp_img, 
                                                            input_dir=args.input_dir, 
                                                            output_dir=args.output_dir) 
    
    records_df['input_file'] = extract_filenames(records_df['input_path'])
    records_df['output_file'] = extract_filenames(records_df['output_path'])

    # Filter out paths that failed
    failed_df = records_df.query('status == "ok"').copy()

    print(f'{len(failed_df)} volumes failed preprocessing pipeline and were removed from final inference.')

    """
    From the extraction data loading. Input CSV needs 'pat_id' column (id from the output of the preprocess) and 'label'(?): 

        def __getitem__(self, idx):
            pat_id = str(self.dataframe.loc[idx, 'pat_id'])
            label = self.dataframe.loc[idx, 'label']  # Regression value for stroke/MCI
            #dataset = str(self.dataframe.loc[idx, 'dataset'])
            
            # Construct image path for MCI/Stroke format
            img_path = os.path.join(self.root_dir,  pat_id  + ".nii.gz")
            sample = {"image": img_path}
            sample = self.transform(sample)
            return {"image": sample["image"], "label": torch.tensor(label, dtype=torch.float32)}

    """

    # Create a dummy column to satisfy extraction
    records_df['label'] = 0

    # Extraction wants 'pat_id'
    records_df = records_df.rename({'id': 'pat_id'}, axis=1)

    """
    !python get_brainiac_features.py \
    --checkpoint ./checkpoints/BrainIAC.ckpt \
    --input_csv ./data/csvs/sample.csv \
    --output_csv ./inference/features/features.csv \
    --root_dir ./data/sample/processed
    """

    # brainiac.extract_features.main(args.input_csv, 
    #                                 args.output_csv, 
    #                                 args.root_dir, 
    #                                 args.checkpoint, 
    #                                 args.batch_size, 
    #                                 args.num_workers)
    
    extract_features.main(records_path, 
                                    args.output_csv, # Final output file path
                                    args.output_dir, # This is the dir with the preprocessed volumes from above
                                    args.checkpoint, # Model checkpoint file
                                    args.batch_size, 
                                    args.num_workers
                                )
    
    if args.input_csv is not None:
        gt_df = pd.read_csv(args.input_csv)
        gt_df['file'] = extract_filenames(gt_df[args.input_path_col])

        # filter to just successful volumes
        records_df = records_df.query('status == "ok"')

        # Merge source information into our new dataframe with processed vol paths
        gt_df = gt_df.merge(
            records_df[['input_path', 'input_file', 'id']],
            left_on='input_file',
            right_on='file',
            how='inner'
        )

        # TODO once we see the output format
        # pred_df = pd.read_csv(args.output_csv)


        # combined_csv_path = Path(args.output_csv).parent / 'tmp_combined.csv'
        # gt_df.to_csv(combined_csv_path, index=False)