if __name__=='__main__':
    try:
        warnings.filewarnings('ignore')
        os.envoron['CUDA_VISIBLE_DEVICES']=''
        model_files_to_be_downloaded=get_models_from_model_yaml()
        if model_files_to_be_downloaded:
            os.makedirs(models_directory,exist_ok=True)
            try:
                with requests.session() as session:
                    for model_file_url in model_files_to_be_downloaded:
                        print(f'downloading {model_file_url.rsplit('/',1)[-1]}..')
                        with session.get(model_file_url,stream=True,verify=False) as response:
                            response.raise_for_status()
                            with open(models_directory_model_file_url.rsplit('/',1)[-1],'wb') as local_file:
                                for chunk in response.iter_content(chunk_size=8192):
                                    local_file.write(chunk)
                            print(f'downloaded {model_file_url.rsplit("/",1)[-1]} successfully')
            except Exception as ex:
                print(model_file_url.rsplit('/',1)[-1]+'not found')
            print('downloaded all models')
            scan_results=[]
            for_,_,model_files in os.walk(models_directory):
            for input_file_path in model_files:
                try:
                    print(f'scanning{input_file_path}')
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        result=subprocess.run(["modelscan","-r","json","-p",models_directory+input_file_path],
                        capture_output=True,text=True)
                        print(f'scan result :{result}')

