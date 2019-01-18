from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from contextlib import ExitStack
import argparse
import json

def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output-file', type=argparse.FileType('w'), help='path to output file')

    parser.add_argument('--batch-size', type=int, default=1, help='The batch size to use for processing')



    args = parser.parse_args()

    return args

def get_predictor():
   
    return Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")


def run(predictor,
        output_file,
        print_to_console
        ):

    def _run_predictor_sentence(sentence):
        value= predictor.predict(sentence)
        print ("sentence prediction",value)


    
    def _run_predictor_batch(batch_data):
        if len(batch_data) == 1:
            result = predictor.predict_json(batch_data[0])
            
            results = [result]
        else:
            results = predictor.predict_batch_json(batch_data)

        for model_input, output in zip(batch_data, results):
            string_output = predictor.dump_line(output)
            if print_to_console:
                print("input: ", model_input)
                print("batch data prediction: ", string_output)
            if output_file:
                output_file.write(string_output)

    sentence= "Anna bought a car from Allen"
    _run_predictor_sentence(sentence)	

    batch_data = [{'sentence': 'Which NFL team represented the AFC at Super Bowl 50?'},{'sentence': 'Where did Super Bowl 50 take place?'},{'sentence': 'Which NFL team won Super Bowl 50?'}]
    _run_predictor_batch(batch_data)


    
def main():
    args = get_arguments()
    predictor = get_predictor()
    output_file = None
    print_to_console = False

    with ExitStack() as stack:
          
        if args.output_file:
            output_file = stack.enter_context(args.output_file)  

        if not args.output_file:
            print_to_console = True

        run(predictor,
            output_file,
            print_to_console
            )

if __name__ == '__main__':
    main()
