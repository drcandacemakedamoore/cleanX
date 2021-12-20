Search.setIndex({docnames:["cleanX","cleanX.cli","cleanX.cli.dataset","cleanX.cli.dicom","cleanX.cli.images","cleanX.cli.main","cleanX.dataset_processing","cleanX.dataset_processing.dataframes","cleanX.dicom_processing","cleanX.dicom_processing.pydicom_adapter","cleanX.dicom_processing.simpleitk_adapter","cleanX.dicom_processing.source","cleanX.image_work","cleanX.image_work.image_functions","cleanX.image_work.journaling_pipeline","cleanX.image_work.pipeline","cleanX.image_work.steps","cli","developers","index","medical-professionals","modules"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,sphinx:56},filenames:["cleanX.rst","cleanX.cli.rst","cleanX.cli.dataset.rst","cleanX.cli.dicom.rst","cleanX.cli.images.rst","cleanX.cli.main.rst","cleanX.dataset_processing.rst","cleanX.dataset_processing.dataframes.rst","cleanX.dicom_processing.rst","cleanX.dicom_processing.pydicom_adapter.rst","cleanX.dicom_processing.simpleitk_adapter.rst","cleanX.dicom_processing.source.rst","cleanX.image_work.rst","cleanX.image_work.image_functions.rst","cleanX.image_work.journaling_pipeline.rst","cleanX.image_work.pipeline.rst","cleanX.image_work.steps.rst","cli.rst","developers.rst","index.rst","medical-professionals.rst","modules.rst"],objects:{"":[[0,0,0,"-","cleanX"]],"cleanX.cli":[[2,0,0,"-","dataset"],[3,0,0,"-","dicom"],[4,0,0,"-","images"],[5,0,0,"-","main"]],"cleanX.cli.dicom":[[3,1,1,"","create_reader"]],"cleanX.cli.main":[[5,2,1,"","Config"]],"cleanX.cli.main.Config":[[5,3,1,"","__init__"],[5,3,1,"","add_setting"],[5,4,1,"","defaults"],[5,3,1,"","get_setting"],[5,3,1,"","merge"],[5,3,1,"","parse"]],"cleanX.dataset_processing":[[7,0,0,"-","dataframes"]],"cleanX.dataset_processing.dataframes":[[7,2,1,"","CSVSource"],[7,2,1,"","ColumnsSource"],[7,2,1,"","DFSource"],[7,5,1,"","GuesserError"],[7,2,1,"","JSONSource"],[7,2,1,"","MLSetup"],[7,2,1,"","MultiSource"],[7,2,1,"","Report"],[7,1,1,"","check_paths_for_group_leakage"],[7,1,1,"","see_part_potential_bias"],[7,1,1,"","show_duplicates"],[7,1,1,"","string_source"],[7,1,1,"","understand_df"]],"cleanX.dataset_processing.dataframes.CSVSource":[[7,3,1,"","__init__"],[7,3,1,"","to_dataframe"]],"cleanX.dataset_processing.dataframes.ColumnsSource":[[7,3,1,"","to_dataframe"]],"cleanX.dataset_processing.dataframes.DFSource":[[7,3,1,"","__init__"],[7,3,1,"","to_dataframe"]],"cleanX.dataset_processing.dataframes.JSONSource":[[7,3,1,"","__init__"],[7,3,1,"","to_dataframe"]],"cleanX.dataset_processing.dataframes.MLSetup":[[7,3,1,"","__init__"],[7,3,1,"","bias"],[7,3,1,"","concat_dataframe"],[7,3,1,"","duplicated"],[7,3,1,"","duplicated_frame"],[7,3,1,"","duplicates"],[7,3,1,"","generate_report"],[7,3,1,"","get_sensitive_list"],[7,3,1,"","get_unique_id"],[7,3,1,"","guess_source"],[7,4,1,"","known_sources"],[7,3,1,"","leakage"],[7,3,1,"","metadata"],[7,3,1,"","pics_in_both_groups"]],"cleanX.dataset_processing.dataframes.MultiSource":[[7,3,1,"","__init__"],[7,3,1,"","to_dataframe"]],"cleanX.dataset_processing.dataframes.Report":[[7,3,1,"","__init__"],[7,3,1,"","report_bias"],[7,3,1,"","report_duplicates"],[7,3,1,"","report_leakage"],[7,3,1,"","report_understand"],[7,3,1,"","subsection_html"],[7,3,1,"","subsection_text"],[7,3,1,"","to_ipwidget"],[7,3,1,"","to_text"]],"cleanX.dicom_processing":[[9,0,0,"-","pydicom_adapter"],[10,0,0,"-","simpleitk_adapter"],[11,0,0,"-","source"]],"cleanX.dicom_processing.pydicom_adapter":[[9,2,1,"","PydicomDicomReader"],[9,1,1,"","get_jpg_with_pydicom"]],"cleanX.dicom_processing.pydicom_adapter.PydicomDicomReader":[[9,3,1,"","__init__"],[9,4,1,"","date_fields"],[9,3,1,"","dicom_date_to_date"],[9,3,1,"","dicom_time_to_time"],[9,4,1,"","exclude_field_types"],[9,4,1,"","exclude_fields"],[9,3,1,"","read"],[9,3,1,"","rip_out_jpgs"],[9,4,1,"","time_fields"]],"cleanX.dicom_processing.simpleitk_adapter":[[10,2,1,"","MetadataHelper"],[10,2,1,"","SimpleITKDicomReader"],[10,1,1,"","rip_out_jpgs_sitk"]],"cleanX.dicom_processing.simpleitk_adapter.MetadataHelper":[[10,3,1,"","__init__"],[10,3,1,"","fetch_image"],[10,3,1,"","fetch_metadata"]],"cleanX.dicom_processing.simpleitk_adapter.SimpleITKDicomReader":[[10,3,1,"","__init__"],[10,4,1,"","date_fields"],[10,3,1,"","dicom_date_to_date"],[10,3,1,"","dicom_time_to_time"],[10,4,1,"","exclude_fields"],[10,3,1,"","read"],[10,3,1,"","rip_out_jpgs"],[10,4,1,"","time_fields"]],"cleanX.dicom_processing.source":[[11,2,1,"","DirectorySource"],[11,2,1,"","GlobSource"],[11,2,1,"","MultiSource"],[11,2,1,"","Source"],[11,1,1,"","rename_file"]],"cleanX.dicom_processing.source.DirectorySource":[[11,3,1,"","__init__"],[11,3,1,"","get_tag"],[11,3,1,"","items"]],"cleanX.dicom_processing.source.GlobSource":[[11,3,1,"","__init__"],[11,3,1,"","get_tag"],[11,3,1,"","items"]],"cleanX.dicom_processing.source.MultiSource":[[11,3,1,"","__init__"],[11,3,1,"","get_tag"],[11,3,1,"","items"]],"cleanX.dicom_processing.source.Source":[[11,3,1,"","get_tag"],[11,3,1,"","items"]],"cleanX.image_work":[[12,1,1,"","create_pipeline"],[13,0,0,"-","image_functions"],[14,0,0,"-","journaling_pipeline"],[15,0,0,"-","pipeline"],[12,1,1,"","restore_pipeline"],[16,0,0,"-","steps"]],"cleanX.image_work.image_functions":[[13,2,1,"","Rotator"],[13,1,1,"","augment_and_move"],[13,1,1,"","avg_image_maker"],[13,1,1,"","avg_image_maker_by_label"],[13,1,1,"","blur_out_edges"],[13,1,1,"","create_matrix"],[13,1,1,"","crop"],[13,1,1,"","crop_np"],[13,1,1,"","crop_pil"],[13,1,1,"","crop_them_all"],[13,1,1,"","dataframe_up_my_pics"],[13,1,1,"","dimensions_to_df"],[13,1,1,"","dimensions_to_histo"],[13,1,1,"","find_big_lines"],[13,1,1,"","find_by_sample_upper"],[13,1,1,"","find_close_images"],[13,1,1,"","find_duplicated_images"],[13,1,1,"","find_duplicated_images_todf"],[13,1,1,"","find_outliers_by_mean_to_df"],[13,1,1,"","find_outliers_by_total_mean"],[13,1,1,"","find_sample_upper_greater_than_lower"],[13,1,1,"","find_suspect_text"],[13,1,1,"","find_suspect_text_by_length"],[13,1,1,"","find_tiny_image_differences"],[13,1,1,"","find_very_hazy"],[13,1,1,"","give_size_count_df"],[13,1,1,"","give_size_counted_dfs"],[13,1,1,"","harsh_sharpie_enhance"],[13,1,1,"","histogram_difference_for_inverts"],[13,1,1,"","histogram_difference_for_inverts_todf"],[13,1,1,"","image_quality_by_size"],[13,1,1,"","make_contour_image"],[13,1,1,"","make_histo_scaled_folder"],[13,1,1,"","proportions_ht_wt_to_histo"],[13,1,1,"","reasonable_rotation_augmentation"],[13,1,1,"","rescale_range_from_histogram_low_end"],[13,1,1,"","salting"],[13,1,1,"","separate_image_averager"],[13,1,1,"","set_image_variability"],[13,1,1,"","show_close_images"],[13,1,1,"","show_images_in_df"],[13,1,1,"","show_major_lines_on_image"],[13,1,1,"","simple_rotation_augmentation"],[13,1,1,"","simple_spinning_template"],[13,1,1,"","subtle_sharpie_enhance"],[13,1,1,"","tesseract_specific"],[13,1,1,"","zero_to_twofivefive_simplest_norming"]],"cleanX.image_work.image_functions.Rotator":[[13,2,1,"","RotationIterator"],[13,3,1,"","__init__"],[13,3,1,"","iter"]],"cleanX.image_work.image_functions.Rotator.RotationIterator":[[13,3,1,"","__init__"],[13,3,1,"","__iter__"]],"cleanX.image_work.journaling_pipeline":[[14,2,1,"","JournalingPipeline"]],"cleanX.image_work.journaling_pipeline.JournalingPipeline":[[14,3,1,"","__init__"],[14,3,1,"","process"],[14,3,1,"","restore"]],"cleanX.image_work.pipeline":[[15,2,1,"","DirectorySource"],[15,2,1,"","GlobSource"],[15,2,1,"","MultiSource"],[15,2,1,"","Pipeline"],[15,5,1,"","PipelineError"]],"cleanX.image_work.pipeline.DirectorySource":[[15,3,1,"","__init__"],[15,3,1,"","__iter__"]],"cleanX.image_work.pipeline.GlobSource":[[15,3,1,"","__init__"],[15,3,1,"","__iter__"]],"cleanX.image_work.pipeline.MultiSource":[[15,3,1,"","__init__"],[15,3,1,"","__iter__"]],"cleanX.image_work.pipeline.Pipeline":[[15,3,1,"","__init__"],[15,3,1,"","process"]],"cleanX.image_work.steps":[[16,2,1,"","Acquire"],[16,2,1,"","Crop"],[16,2,1,"","HistogramNormalize"],[16,2,1,"","Normalize"],[16,2,1,"","RegisteredStep"],[16,2,1,"","Save"],[16,2,1,"","Step"],[16,1,1,"","get_known_steps"]],"cleanX.image_work.steps.Acquire":[[16,3,1,"","__init__"],[16,3,1,"","__reduce__"],[16,3,1,"","apply"],[16,3,1,"","from_cmd_args"],[16,3,1,"","read"],[16,3,1,"","to_json"],[16,3,1,"","write"]],"cleanX.image_work.steps.Crop":[[16,3,1,"","__init__"],[16,3,1,"","__reduce__"],[16,3,1,"","apply"],[16,3,1,"","from_cmd_args"],[16,3,1,"","read"],[16,3,1,"","to_json"],[16,3,1,"","write"]],"cleanX.image_work.steps.HistogramNormalize":[[16,3,1,"","__init__"],[16,3,1,"","__reduce__"],[16,3,1,"","apply"],[16,3,1,"","from_cmd_args"],[16,3,1,"","read"],[16,3,1,"","to_json"],[16,3,1,"","write"]],"cleanX.image_work.steps.Normalize":[[16,3,1,"","__init__"],[16,3,1,"","__reduce__"],[16,3,1,"","apply"],[16,3,1,"","from_cmd_args"],[16,3,1,"","read"],[16,3,1,"","to_json"],[16,3,1,"","write"]],"cleanX.image_work.steps.RegisteredStep":[[16,3,1,"","__init__"],[16,3,1,"","mro"]],"cleanX.image_work.steps.Save":[[16,3,1,"","__init__"],[16,3,1,"","__reduce__"],[16,3,1,"","apply"],[16,3,1,"","from_cmd_args"],[16,3,1,"","read"],[16,3,1,"","to_json"],[16,3,1,"","write"]],"cleanX.image_work.steps.Step":[[16,3,1,"","__init__"],[16,3,1,"","__reduce__"],[16,3,1,"","apply"],[16,3,1,"","from_cmd_args"],[16,3,1,"","read"],[16,3,1,"","to_json"],[16,3,1,"","write"]],"python3--m-cleanX":[[17,6,1,"cmdoption-python3-m-cleanX-c","--config"],[17,6,1,"cmdoption-python3-m-cleanX-f","--config-file"],[17,6,1,"cmdoption-python3-m-cleanX-v","--verbosity"],[17,6,1,"cmdoption-python3-m-cleanX-c","-c"],[17,6,1,"cmdoption-python3-m-cleanX-f","-f"],[17,6,1,"cmdoption-python3-m-cleanX-v","-v"]],"python3--m-cleanX-dataset-report":[[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-l","--label-tag"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-report-bias","--no-report-bias"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-report-duplicates","--no-report-duplicates"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-report-leakage","--no-report-leakage"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-report-understand","--no-report-understand"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-o","--output"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-report-bias","--report-bias"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-report-duplicates","--report-duplicates"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-report-leakage","--report-leakage"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-report-understand","--report-understand"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-s","--sensitive-category"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-t","--test-source"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-r","--train-source"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-i","--unique_id"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-i","-i"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-l","-l"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-o","-o"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-r","-r"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-s","-s"],[17,6,1,"cmdoption-python3-m-cleanX-dataset-report-t","-t"]],"python3--m-cleanX-dicom-extract":[[17,6,1,"cmdoption-python3-m-cleanX-dicom-extract-c","--config-reader"],[17,6,1,"cmdoption-python3-m-cleanX-dicom-extract-i","--input"],[17,6,1,"cmdoption-python3-m-cleanX-dicom-extract-o","--output"],[17,6,1,"cmdoption-python3-m-cleanX-dicom-extract-c","-c"],[17,6,1,"cmdoption-python3-m-cleanX-dicom-extract-i","-i"],[17,6,1,"cmdoption-python3-m-cleanX-dicom-extract-o","-o"]],"python3--m-cleanX-dicom-report":[[17,6,1,"cmdoption-python3-m-cleanX-dicom-report-c","--config-reader"],[17,6,1,"cmdoption-python3-m-cleanX-dicom-report-i","--input"],[17,6,1,"cmdoption-python3-m-cleanX-dicom-report-o","--output"],[17,6,1,"cmdoption-python3-m-cleanX-dicom-report-c","-c"],[17,6,1,"cmdoption-python3-m-cleanX-dicom-report-i","-i"],[17,6,1,"cmdoption-python3-m-cleanX-dicom-report-o","-o"]],"python3--m-cleanX-images-restore-pipeline":[[17,6,1,"cmdoption-python3-m-cleanX-images-restore-pipeline-j","--journal-dir"],[17,6,1,"cmdoption-python3-m-cleanX-images-restore-pipeline-s","--skip"],[17,6,1,"cmdoption-python3-m-cleanX-images-restore-pipeline-r","--source"],[17,6,1,"cmdoption-python3-m-cleanX-images-restore-pipeline-j","-j"],[17,6,1,"cmdoption-python3-m-cleanX-images-restore-pipeline-r","-r"],[17,6,1,"cmdoption-python3-m-cleanX-images-restore-pipeline-s","-s"]],"python3--m-cleanX-images-run-pipeline":[[17,6,1,"cmdoption-python3-m-cleanX-images-run-pipeline-b","--batch-size"],[17,6,1,"cmdoption-python3-m-cleanX-images-run-pipeline-j","--journal"],[17,6,1,"cmdoption-python3-m-cleanX-images-run-pipeline-k","--keep-journal"],[17,6,1,"cmdoption-python3-m-cleanX-images-run-pipeline-r","--source"],[17,6,1,"cmdoption-python3-m-cleanX-images-run-pipeline-s","--step"],[17,6,1,"cmdoption-python3-m-cleanX-images-run-pipeline-b","-b"],[17,6,1,"cmdoption-python3-m-cleanX-images-run-pipeline-j","-j"],[17,6,1,"cmdoption-python3-m-cleanX-images-run-pipeline-k","-k"],[17,6,1,"cmdoption-python3-m-cleanX-images-run-pipeline-r","-r"],[17,6,1,"cmdoption-python3-m-cleanX-images-run-pipeline-s","-s"]],cleanX:[[1,0,0,"-","cli"],[6,0,0,"-","dataset_processing"],[8,0,0,"-","dicom_processing"],[12,0,0,"-","image_work"]]},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"],"5":["py","exception","Python exception"],"6":["std","cmdoption","program option"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:attribute","5":"py:exception","6":"std:cmdoption"},terms:{"0":[10,12,13,14,16],"000":20,"048":20,"1":[13,20],"10":13,"100":20,"2":[7,18],"20":13,"2000":13,"2048":13,"2500":13,"255":[13,16],"2d":13,"3":[18,20],"36":13,"360":13,"37":18,"5":[13,16],"500":13,"576":20,"7":[13,18],"8":13,"999":13,"abstract":[7,11],"byte":[7,9,15],"case":[7,13,15,17,18],"catch":13,"class":[5,7,9,10,11,13,14,15,16,19,20],"default":[5,9,10,13,16,17],"do":[9,10,11,13,16,20],"final":13,"float":13,"function":[5,7,9,10,11,13,16,17,19,20],"import":[7,13],"int":[7,12,13,15],"long":18,"new":[5,7,9,13,20],"null":7,"public":20,"return":[5,7,9,10,11,12,13,14,16],"super":13,"throw":[9,10,16],"true":[5,7,11,12,13,14,15],"try":[7,18,20],A:[7,9,10,11,12,13,15],As:18,At:20,But:20,For:[9,10,13,17,18,19,20],IT:18,If:[7,9,10,12,13,14,15,16,17,18,19,20],In:[13,18],It:[1,7,10,11,13,15,16,20],Such:20,The:[1,5,7,9,10,11,12,13,14,15,16,17,20],There:[18,20],These:[7,9,15,17,18],To:[13,18,20],Will:16,__init__:[5,7,9,10,11,13,14,15,16],__iter__:[13,15],__reduce__:16,abc:[7,11],abdomen:13,abdomin:13,abil:[14,18],abl:18,abnorm:13,about:[7,13,17,19,20],abov:[13,18],abstractmethod:7,accept:18,access:[7,20],accident:7,accomplish:20,accord:[7,20],accross:18,accur:7,accuraci:13,acquir:16,across:20,action:18,activ:18,actual:[16,20],ad:[18,20],add:[7,13,16,18,19],add_set:5,addit:[9,14],addition:20,adress:13,advanc:13,affect:7,after:[12,13,14,17],afterward:18,ag:7,again:[12,14],against:18,aggreg:7,agnost:11,ai:20,aid:[11,20],aim:18,air:13,algorithm:[19,20],all:[7,13,15,16,17,18,20],alli:20,allow:[7,9,13,16],alreadi:[7,18],also:[10,13,18,19,20],altern:18,among:20,amount:20,an:[7,9,10,11,13,14,15,16,18,20],anaconda:18,anaconda_gen_env:18,analysi:13,angl:13,angle1:13,angle2:13,angle_list1:13,angle_start:13,angle_stop:13,ani:[7,9,13,15,20],anonym:20,anoth:18,anyon:20,anyth:13,anywai:13,appear:7,append:15,appli:[7,13,16,18,20],applic:[7,13],approach:18,appropri:20,approxim:13,ar:[5,7,9,12,13,15,16,17,18,19,20],archiv:18,area:[13,20],arg:[17,18],argument:[7,12,14,15,17,18],around:13,arrai:[10,13,16],art:20,artifact:18,ask:[13,18],aspect:[7,13,18],ass:7,asses:7,asset:7,assum:13,attempt:[12,14],attemt:15,attent:13,attribut:18,augment:[13,19,20],augment_and_mov:13,autom:18,automat:20,avail:[1,5,7,16],averag:13,avg_image_mak:13,avg_image_maker_by_label:13,avoid:[15,18,20],awai:13,axesimag:13,b:[13,17],back:13,background:13,bad:18,base:[5,7,9,10,11,13,14,15,16,18],basi:19,basic:20,batch:17,batch_siz:[12,14,15,17],becaus:13,becom:13,been:[7,12,13],befor:[12,14,17,20],begin:20,behavior:14,being:[7,15,17,18,20],believ:18,benefit:20,besid:13,best:18,better:18,between:[13,18],bia:[7,17],big:20,big_siz:13,bigger:13,bin:[13,18],bins_count:13,biolog:13,biomed:20,bit:18,bizarr:13,black:[13,16],blend:13,blue:13,blur:13,blur_out_edg:13,blurred_edge_imag:13,bool:[7,9,12,14],both:[5,7,13,20],branch:18,brand:20,brief:20,bring:20,broader:20,build:[18,19],builder:15,built:7,builtin:17,c:[13,17],cache_dir:16,calcul:7,call:[11,13,16],caller:11,can:[5,7,10,12,13,15,17,18,19,20],cannot:7,capac:18,capit:20,captur:13,care:13,categor:13,categori:[7,17],caught:13,center:13,certain:[13,19,20],cfg:3,chain:[7,15],chanc:13,chang:18,check:[7,11,18,20],check_paths_for_group_leakag:7,chest:[7,13,20],choic:18,chose:20,chosen:17,chould:13,ci:18,classic:13,classmethod:[14,16],clean:[7,13,19,20],cleanx:18,cli:[0,19,21],clsdict:16,cmd_arg:16,code:[7,11,12,14,19,20],coder:20,collect:[7,13],colum:[7,11],column:[7,11,13,17],columnsourc:7,columnssourc:7,combin:[7,18,20],come:20,comfort:20,command:[5,18],commit:18,common:18,compar:13,comparison:13,compat:[9,10,18],complet:[12,14,18],complex:18,compress:13,compression_level:13,comput:[13,20],concat_datafram:7,concatend:7,concurr:[12,17],conda:18,config:[5,17],config_fil:17,config_read:[3,17],configur:[5,7,17],configuraiton:17,consid:[13,17],constitut:7,contain:[7,9,12,14,17,20],content:[10,21],contentd:9,contenttim:9,context:[7,16],contigu:13,contour:13,contribut:20,control:[5,12,14,15,17,18],conveni:18,conver:7,convert:7,cooper:20,coordin:13,copi:13,copy_imag:13,core:[7,13,18],coron:20,could:[13,18],count:[7,13],countri:20,cours:18,creat:[7,12,13,14,15,18,20],create_matrix:13,create_pipelin:12,create_read:3,creation:13,critic:[17,20],crop:[13,16],crop_np:13,crop_pil:13,crop_them_al:13,cross:18,csv:[7,17,20],csvsourc:7,ct:20,curat:20,curios:18,current:[5,18],custom:16,cut:13,cv2:13,cxr:13,d:13,dashboard:18,data:[7,9,10,13,16,17,19,20],data_process:20,databas:[12,14],datafram:[0,6,9,10,11,13],dataframe_image_column:13,dataframe_label_column:13,dataframe_up_my_p:13,datasest:7,dataset:[0,1,7,13,20],dataset_process:[0,21],datastructur:7,date:[9,10],date_field:[9,10],datetim:[9,10],dcm:[9,10],deal:7,debug:17,decid:18,deepli:7,default_el:13,defer:18,defin:13,degre:13,demo:20,depart:18,depend:[7,13,15,17,18],deploi:18,descend:[7,12],describ:[17,18],descript:17,design:[18,20],desir:[13,20],destin:[9,10],detail:7,dev:18,develop:20,df:[7,13],dfsourc:7,diagnos:7,diagnosi:[7,17],diagnosis_str:13,dicom:[0,1,5,9,10,11,18,19,20],dicom_date_to_d:[9,10],dicom_fil:10,dicom_folder_path:9,dicom_process:[0,21],dicom_time_to_tim:[9,10],dicomfile_directori:[9,10],dict:10,dictionari:5,did:[7,18],didn:[12,18],differ:[7,13,16,18],difficult:[9,18],dimens:13,dimension:19,dimensions_to_df:13,dimensions_to_histo:13,dir:17,directori:[5,9,10,11,12,13,14,15,17,18],directorysourc:[11,15],discard:13,discov:7,disk:15,distribut:[7,13,18],divid:13,doctor:20,document:[1,18],doe:13,doesn:[11,15,18],don:18,done:18,down:[13,18,20],download:18,downsid:18,dramat:13,draw:13,ducplic:[7,17],duplic:[7,13,17],duplicated_fram:7,dure:[16,18],e:[7,13,18,20],each:[7,9,10,13,20],earli:7,eas:20,easi:18,easili:20,edg:13,edit:18,eg:[17,18],egg:18,either:[5,7,11,13,18],element:[7,11,13,16],elo:13,email:18,encount:[15,18],end:13,english:13,enough:10,ensur:11,entir:7,env:18,equal:13,error:[7,9,10,13,15,16,17],especi:18,essenti:[18,20],etc:[7,17],ethnic:[7,17],even:20,eventu:18,everi:[13,18],everyon:20,everywher:[18,20],exact:[13,20],exampl:[13,18,20],excel:20,except:[7,15,16],exclud:[9,10],exclude_field:[9,10],exclude_field_typ:9,execut:[12,14,15,16,17],exhibit:17,exist:[5,7],exp:11,expect:[11,16,20],expens:20,experi:[19,20],experiment:18,expertis:20,explan:[9,10],explor:7,exploratori:13,express:[7,15,17],ext:11,extend:[9,10,11,14,16],extens:[7,15,16,17,18],extern:7,extra:5,extract:[7,9,10,11,20],exxtrem:16,ey:13,f:[17,18],fact:7,factor:7,factori:7,fail:[12,14],fair:7,fall:13,fals:[5,12,14,15],falsi:12,familiar:20,far:18,fast:[12,14],fatal:17,featur:7,fed:11,feed:20,fetch:10,fetch_imag:10,fetch_metadata:10,few:20,fewer:7,field:[9,10,20],figur:7,file:[5,7,9,10,11,14,15,16,17,18,19,20],filesystem:15,filter:9,final_avg:13,final_diff:13,find:[7,11,13,18],find_big_lin:13,find_by_sample_upp:13,find_close_imag:13,find_duplicated_imag:13,find_duplicated_images_todf:13,find_outliers_by_mean_to_df:13,find_outliers_by_total_mean:13,find_sample_upper_greater_than_low:13,find_suspect_text:13,find_suspect_text_by_length:13,find_tiny_image_differ:13,find_very_hazi:13,fine:18,finish:17,first:[7,9,10,11,13,16,17],fix:13,flag:[7,10],fle:15,flip:13,fluoroscopi:13,fold:13,folder:[9,13,20],folder_nam:13,follow:[13,20],forc:13,form:10,formal:7,format:[7,16,17,18],forward:[12,14],foulder:13,found:[5,7,13,18,19],fragment:15,frame:[7,13,16],free:20,freeli:20,fresh:[12,14],from:[5,7,9,10,11,12,13,14,16,17,18,20],from_cmd_arg:16,frontal:13,full:13,funuct:[9,10],g:[7,13,20],garbag:20,gender:[7,17],gener:[1,7,9,10,11,13,18,19],generate_report:7,genrat:[9,10],get:[7,10,16,20],get_jpg_with_pydicom:9,get_known_step:16,get_sensitive_list:7,get_set:5,get_tag:11,get_unique_id:7,git:18,github:18,give:[7,9,13],give_size_count_df:13,give_size_counted_df:13,given:[5,7,10,13,15,17,19],glob:[5,13,15,17],glob_is_recurs:[5,17],globsourc:[11,15],go:13,goal:18,good:20,great:20,greater:13,greys_templ:13,ground:18,group:[7,13,20],guess_sourc:7,guessererror:7,ha:[7,13,16,17,18],handi:13,handl:[7,13,20],happen:18,harsh_sharpie_enh:13,hash:18,have:[7,9,10,12,13,18,20],hazi:13,health:20,height:13,help:[7,9,10,13,19,20],helper:[7,10,11,16],here:[1,13,19,20],heurist:7,high:13,highest:13,highli:18,histo:13,histo_ht_wt:13,histo_ht_wt_p:13,histogram:[13,16],histogram_difference_for_invert:13,histogram_difference_for_inverts_todf:13,histogramnorm:16,hold:13,holland:20,home:[5,20],hood:18,hospit:20,hous:20,how:[5,7,12,13,17,20],howev:[13,15,18,20],html:7,human:[7,13,20],i:[7,13,17,20],id:[7,17],idea:19,ideal:20,identifi:7,im:13,imag:[0,1,7,9,10,13,14,15,16,19,20],image_arrai:13,image_data:16,image_directori:13,image_fold:13,image_funct:[0,12],image_quality_by_s:13,image_work:[0,21],imagefileread:10,img:13,img_nam:13,img_pi:13,imgs_fold:13,impercept:13,implement:[7,9,10,11,13,15,16],impli:13,imread:13,inappropri:18,includ:[7,9,13,19,20],inclus:[13,20],incompat:18,incorpor:18,incorrect:20,increment:13,indent:7,index:19,indic:[7,13,20],individu:[11,13],inequ:20,infer:17,info:17,inform:[7,9,13,17,19],inherit:11,initi:[5,7,9,10,13,14,15,17],input:[13,17],insert:11,insid:13,inspect:[7,10],instal:[7,18],instaleld:18,install_dev:18,instanc:[7,10,12,13,14,18],instead:20,institut:20,intend:[7,16,18],interest:7,interfac:[13,15],intermedi:15,intern:[7,9,18],interpret:[5,7,9,10,15,17,18],interrupt:12,introductori:20,invent:18,inversu:13,invert:[13,20],investig:20,isn:[9,10,18],item:11,iter:[7,10,13,14,15],iter_ob:13,iteratbl:13,itertool:[7,15],its:[10,12,15,16],itself:[13,18],j:17,jenkin:18,job:18,join:[9,10,12,13,14,15],journal:[5,12,14,17],journal_dir:[12,14,17],journal_hom:[5,17],journaling_pipelin:[0,12],journalingpipelin:[12,14],jpeg:[9,10,13],jpg:[9,10,13,15,16,20],jpg_folder_path:9,json:[5,7,17],jsonsourc:7,jupyt:[7,20],just:[9,10,18],justif:18,k:[5,17,18],kaggl:20,keep:[12,17],keep_journ:[12,14],kei:5,kept:[12,14],kind:7,knife:13,know:[12,13,14,18,20],known_sourc:7,l:[13,17],label:[7,11,13,17],label_for_haz:13,label_tag:[7,17],label_word:13,lambda:7,last:[12,13,14],later:[7,11,13],layer:[10,18],leak:7,leakag:[7,17],learn:[7,13,17,19,20],least:[13,18],left:13,length:[7,13],length_nam:13,less:[7,19],let:18,level:[7,13,17],librari:[7,9,10,11,13],licens:18,lifecycl:18,like:[7,11,13,17,18,20],limit:18,line:[5,13,20],line_length:13,lingual:13,link:20,lint:18,linter:18,linux:18,list:[7,10,13,15,17],ll:18,load:7,locat:[13,17],lock:20,log:17,longer:18,look:[10,13,15,17,18,19],lot:[18,20],love:9,low:[13,18],lower:[13,19,20],lowest:13,m:18,machin:[7,13,17,19,20],made:[13,18,20],mai:[7,11,12,13,16,17,18,20],main:[0,1],maintain:18,make:[5,7,13,16,18],make_contour_imag:13,make_histo_scaled_fold:13,malawi:20,manag:18,mani:[12,13,14,15,17,20],manipul:13,manufactur:13,map:7,margin:13,mark:13,markup:7,master_df:13,match:[13,15,17],matplotlib:[10,13],matric:13,matrix:13,matter:20,mayb:13,mean:[13,15,18,20],meant:20,memori:[13,15],merg:[5,18],metadata:[7,9,10,11,20],metadatahelp:10,method:[7,9,10,11,13,16,19],microsoft:20,middl:[13,18],might:7,minim:13,mislabel:20,mismatch:13,ml:7,mlsetup:7,modifi:[16,17],modul:[19,20,21],more:[7,13,18,20],most:[13,18,20],mostli:7,move:13,mro:16,mse:13,much:[13,20],multi:13,multipl:[7,15,18],multiprocess:16,multisourc:[7,11,15],multiv:9,multivalu:9,must:[7,12,13,14,15,16,17,18],my_academia:13,n:[13,18],name:[5,7,9,10,11,13,14,15,16,17,18],nation:20,navig:18,nb:13,ndarrai:[10,13,16],near_dup:13,nearli:13,neatli:7,necessari:[5,7,11,17],neck:13,need:[7,9,10,11,16,17,18,20],neg:13,nest:7,net:[7,13],neural:[7,13],new_image_arrai:13,next:16,nois:13,non:[13,20],none:[5,7,9,10,11,12,13,14,15,16],nor:18,normal:[13,16,18],note:[13,18,20],notebook:[7,20],noth:13,notset:17,now:19,np:13,number:[7,13,15,17],number_slic:13,numer:13,numpi:[10,13,16],o:17,object:[5,7,9,10,11,12,13,14,15,16],obtain:[12,16],obviou:18,obvious:[13,20],occasion:20,off:[13,16],often:20,old:20,older:13,onc:[7,15,18],one:[7,9,10,11,12,13,16,18,20],onli:[5,7,9,10,13,16,18,20],op:7,open:[5,7,20],opencv2:13,opencv:13,oper:[18,20],option:[7,9,11,17,18],order:[15,16,18],org:18,organ:[7,18],organzi:18,orient:19,origin:[11,13,18],origin_fold:13,os:[7,9,10,12,13,14,15,18],other:[7,9,13,19,20],otherwis:[7,12,13,14],our:[18,19,20],out:[7,9,10,11,13,15,18,19,20],outcom:20,outlier:13,outlin:13,output:[7,13,17,20],output_directori:[9,10],outsid:[18,20],over:[13,20],overrepres:7,overrid:[5,10,12,14,16],own:11,p:12,pa:13,packag:[11,18,19,20,21],page:[18,19],pair:17,panda:[7,11,13],parallel:15,param:13,paramet:[5,7,9,10,11,12,13,14,15,16],pars:[5,9,10,17],parser:[5,17],part:[7,19],partial:13,pass:[7,12,14,15,17,18],patch:18,path:[5,7,9,10,11,12,13,14,15,16,17],pathlib:7,patholog:[7,13],patient:[7,13,17,18],pattern:[5,15,17],pd_arg:7,peopl:20,pep8:18,per:13,percent:13,percent_height_of_sampl:13,percentag:13,percentage_to_say_outli:13,percentil:13,perfect:13,persist:16,pic_nam:13,pici:13,pick:[12,13,18],pickl:16,pics_in_both_group:7,pictur:13,pil:13,pillow:13,pip:18,pipelin:[0,5,12,14],pipelineerror:15,pixel:[10,13],place:[9,10,13,17,20],plain:7,platform:18,pleas:18,plot:13,plot_limit:13,pneumonia:[7,13],point:20,poorer:20,popul:13,portabl:18,possibl:[15,16,17,18],potenti:[13,17],power:20,practic:[18,20],precis:13,preconfigur:[12,14],preferred_dicom_pars:5,prepar:20,present:20,presist:14,presum:13,previou:16,previous:[12,14],previv:16,print:[7,13,17],prioriti:18,privat:18,probabl:18,problem:[7,13,18,20],problemat:20,proces:16,process:[7,11,12,14,15,16,17,18,20],produc:[7,9,11,13,18],product:[18,20],profil:18,program:20,programm:[18,19,20],progress:14,project:[18,20],properi:17,properti:[7,17],proport:13,proportions_ht_wt_to_histo:13,proprietari:20,protocol:13,prototyp:13,provid:[7,11,17,18,20],publish:18,pull:18,pure:20,push:18,put:[7,9,10,13],py:18,pydicom:[5,9,11],pydicom_adapt:[0,8],pydicomdicomread:9,pytest:18,python3:18,python:[9,10,17],python_vers:18,pyton:10,q:13,qualiti:[13,19,20],question:[13,20],quickli:13,quit:18,r:[13,17],race:7,radiolog:[7,13,20],radiologist:20,rai:[7,13,20],rais:[7,16],random_within_domain:13,rang:13,rare:18,rather:7,ratio:13,raw_src:7,read:[5,7,9,10,11,13,15,16,18,20],read_csv:7,read_json:7,reader:[9,10,11,17,20],readi:[7,18],real:13,realist:13,realiti:20,realli:13,reason:[9,13,20],reasonable_rotation_augment:13,rebas:18,recogn:5,rectangl:13,recurs:[7,11,15,17],redad:10,ref_ms:13,refer:20,registeredstep:16,regradless:18,regular:[7,13],relat:[7,18],releas:18,relev:7,reli:[7,18,20],remain:14,rememb:7,remov:18,rename_fil:11,rent:18,repeat:17,replac:5,repo:18,report:[7,15,18],report_bia:7,report_dupl:7,report_leakag:7,report_understand:7,repres:[7,9,10],represent:[7,9,10],reproduc:18,request:18,requir:[17,18],resampl:13,rescale_range_from_histogram_low_end:13,research:[9,10,20],resolut:16,rest:[12,14],restor:[12,14],restore_pipelin:12,result:[7,11,13,15,16],resum:[12,14,17],reveal:[13,20],rid:20,right:13,rip_out_jpg:[9,10],rip_out_jpgs_sitk:10,risk:20,rotat:13,rotationiter:13,row:[7,20],run:[5,7,13,15,18,20],runner:5,runnig:18,runtimeerror:15,s:[7,9,10,13,16,17],sai:20,salient:13,salt:13,same:[7,13,20],sampl:[7,13],satisfi:18,save:[7,9,10,15,16],scale:13,school:20,screen:7,script:18,search:[7,19],second:[7,11,16,17],section:[7,18],see:[9,10,13,14,17,20],see_part_potential_bia:7,seem:18,select:[7,9,17,18],senist:7,sens:5,sensit:[7,15,17],sensitive_categori:17,sensitive_column_list:7,sensitive_list:7,sent:13,separ:[1,7],separate_image_averag:13,sequenc:[7,9,12,15],seri:[9,10,20],serial:16,seriesd:9,server:18,set:[5,7,12,13,17,20],set_image_vari:13,set_of_imag:13,setup:[7,18],setuptool:18,sever:[18,20],share:20,sharpen:13,sharper:13,should:[1,5,7,9,10,11,12,13,15,16,17,18,20],show:[13,20],show_close_imag:13,show_dupl:7,show_images_in_df:13,show_major_lines_on_imag:13,side:[11,13],similar:[7,13,15,17,18,20],similarli:18,simpl:[7,16],simple_rotation_augment:13,simple_spinning_templ:13,simpleitk:[5,10,11],simpleitk_adapt:[0,8],simpleitkdicomread:10,simpli:20,sinc:18,singl:[7,13,18],situ:13,situat:18,size:[13,17],skip:[12,14,17],slice:[13,20],smaller:13,so:[13,20],solut:20,solv:20,some:[7,9,13,18,19,20],somewher:16,sort:[7,13],sourc:[0,5,7,8,9,10,13,14,15,17,18,20],source_directori:13,spars:13,special:20,specif:[13,19,20],specifi:[5,7,12,13,15,17],specific_imag:13,spend:20,sphinx_click:1,spike:13,spin:13,splitext:7,squar:13,start:[13,14,15,18],state:[14,20],stdout:17,step:[0,12,13,14,15,17],still:20,stop:13,storag:14,store:[5,9,10,11,12,14,17],str:[5,7,9,10,12,13,14,15],straight:13,string:[7,9,10,13],string_sourc:7,stringent:13,structur:9,studi:[7,10],studyd:9,studytim:9,stymi:20,subject:13,submiss:18,submodul:[0,21],subpackag:[19,21],subsect:7,subsection_html:7,subsection_text:7,substanti:20,subtle_sharpie_enh:13,subtli:13,subtract:13,succe:18,success:[12,14,16],suggest:[13,20],suitabl:[5,7,11,12,13,14,16,17,18],summar:7,sup:13,superclass:7,suppli:[7,11],support:[17,18],suppos:18,suspect:[7,13],synthet:13,system:18,t:[9,10,11,12,15,16,17,18],tab_fight_bias2:7,tabul:7,tabular:20,tad:13,tag:[9,11,13,17,18],tail_cut_perc:[13,16],take:[7,11,13,17,18],target:[11,13,16],target_fold:13,target_nam:13,task:[17,18,20],taught:18,technic:[13,20],techniqu:13,technolog:20,tell:18,templat:13,term:[7,13,20],tesseract_specif:13,test:[7,13,17],test_df:7,test_sourc:17,test_src:7,text:[7,13],than:[7,13,18],thei:[5,7,9,10,13,18],them:[9,10,13,18],themselv:13,therefor:[13,16,18,20],thi:[5,7,9,10,11,12,13,14,15,16,17,18,19,20],thing:18,think:7,those:[5,7,9,11,12,13,14,18],though:18,three:20,threshold4:13,thrill:20,through:[7,18],thrown:20,thu:13,time:[9,10,13,17,18,20],time_field:[9,10],tini:13,titl:13,to_datafram:7,to_ipwidget:7,to_json:16,to_list:13,to_text:7,togeth:7,too:[13,18],took:20,tool:20,top:[13,20],touch:20,toward:7,train:[7,17],train_df:7,train_sourc:17,train_src:7,transform:[11,13,16],translat:[9,10,20],tri:7,tupl:[7,11,13,16],turn:7,twist:13,two:[7,11,14,16,17,19],txt:[17,18],typcial:7,type:[7,9,10,11,12,13,14,16,17,19,20],typeerror:7,typic:[7,13,17,18],unclean:19,unclear:13,under:[13,18],understand:[7,17],understand_df:7,understnd:13,unfortun:[18,20],union:[7,12,13,14,15],uniqu:[7,13,17],unique_id:[7,17],unless:16,unreli:18,unusu:18,unzip:18,up:[12,13],upper:13,upsid:[13,20],us:[1,5,7,10,12,13,14,15,16,17,18,19,20],usabl:7,usag:13,user:[9,10,13],usual:[7,9,13,17,18],utah:20,util:[7,9,10],v1:18,v:[5,17,18],valid:15,valu:[5,7,10,11,13,16,17],value_for_lin:13,vanilla:18,vari:[13,18],variabl:[5,13],variou:[5,7,13],venv37:18,venv3:18,venv:18,verbos:17,veri:[13,18,20],version:[13,18],video:20,view:13,virabl:13,virtual:18,vision:20,wa:[7,16,18,19,20],wai:[7,12,13,15],want:[11,12,14,18,20],warn:17,wasn:16,we:[13,19,20],websit:19,well:[13,20],were:[7,13],what:[7,9,13,18],when:[5,7,13,15,16,20],where:[5,9,10,13,17,18],whether:[7,12,14,15,17],which:[5,7,10,13,15,18,20],whichev:[7,20],who:[11,20],whole:20,whose:5,why:20,widget:7,width:13,wiki:19,within:13,without:20,women:7,wonder:20,word:13,work:[13,19,20],workflow:18,worklfow:18,world:13,worsen:20,would:[7,13,18,19,20],wrapper:13,write:[15,16],x:[7,13,20],xlsx:20,yet:[9,10,18],yield:[11,14,15],yml:18,you:[7,9,10,11,12,13,14,15,16,17,18,19,20],your:[7,9,11,13,19],zero:13,zero_to_twofivefive_simplest_norm:13},titles:["cleanX package","cleanX.cli package","cleanX.cli.dataset module","cleanX.cli.dicom module","cleanX.cli.images module","cleanX.cli.main module","cleanX.dataset_processing package","cleanX.dataset_processing.dataframes module","cleanX.dicom_processing package","cleanX.dicom_processing.pydicom_adapter module","cleanX.dicom_processing.simpleitk_adapter module","cleanX.dicom_processing.source module","cleanX.image_work package","cleanX.image_work.image_functions module","cleanX.image_work.journaling_pipeline module","cleanX.image_work.pipeline module","cleanX.image_work.steps module","cleanX Command-Line Interface","Developer\u2019s Guide","cleanX documentation","Medical Professional Documentation","cleanX"],titleterms:{"do":18,"import":20,It:18,The:[18,19],api:19,cleanx:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21],cli:[1,2,3,4,5],code:18,command:[17,19],content:[0,1,6,8,12,19],continu:18,datafram:7,dataset:[2,17],dataset_process:[6,7],develop:[18,19],dicom:[3,17],dicom_process:[8,9,10,11],differ:20,document:[19,20],environ:18,extract:17,guid:[18,19],imag:[4,17],image_funct:13,image_work:[12,13,14,15,16],indic:19,integr:18,interfac:[17,19],introduct:20,journaling_pipelin:14,line:[17,19],m:17,main:5,make:20,medic:[19,20],modul:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],packag:[0,1,6,8,12],pipelin:[15,17],profession:[19,20],pydicom_adapt:9,python3:17,python:18,rational:18,report:17,restor:17,run:17,s:[18,19],send:18,set:18,simpleitk_adapt:10,sourc:11,step:16,style:18,submodul:[1,6,8,12],subpackag:0,tabl:19,test:18,tradit:18,up:18,wai:18,we:18,what:20,work:18,workflow:20,your:18}})