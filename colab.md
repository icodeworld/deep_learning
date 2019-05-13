# colab使用谷歌云中文件

默认是不能读入google drive的数据的，每次都要重新上传，费时费力。所以这篇博客是让colab用户能够使用google drive的工作文件夹

### step1

首先需要让colab获得google drive的授权，但需要先装包：

```
#装opam,后装google-drive-ocamlfuse
!apt-get install opam
!opam init
!opam update
!opam install depext
!opam depext google-drive-ocamlfuse
!opam install google-drive-ocamlfuse
#进行授权操作
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!/root/.opam/system/bin/google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | /root/.opam/system/bin/google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
#!!!注意，里面的/root/.opam/system/bin/google-drive-ocamlfuse换成你自己的路径,一般来说你也会得到和我一样的结果
# 指定Google Drive云端硬盘的根目录，名为drive
!mkdir -p drive
!/root/.opam/system/bin/google-drive-ocamlfuse drive

```

按提示执行

### step 2

确认是否成功

```python
!ls 
```

此时colab中出现drive的文件夹，里面就是你的google drive的根目录文件

### step3

想执行哪个文件夹下的文件，更换执行的工作文件夹即可。如

```
import os
os.chdir("drive/Colab Notebooks") 
```

参考链接：https://www.jianshu.com/p/1c1f47748827
