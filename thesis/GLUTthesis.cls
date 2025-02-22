%!Mode:: "TeX:UTF-8"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 桂林理工大学本科生学位论文 LaTeX 模版
%% 2023-02-01 v1.1
%% 作者：胡光辉
%% E-mail: guanghui.hu@me.com
%% 本模版基于 CSUthesis.cls 修改
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 本模版根据桂林理工大学本科生学位论文撰写规范创建
%% 论文内容一般应由9个主要部分组成，依次为：
%% 1.封面
%% 2.独创性声明
%% 3.中文摘要；
%% 4.英文摘要；
%% 5.目录；
%% 6.论文正文；
%% 7.参考文献；
%% 8.致谢
%% 9.附录（必要时）。
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{GLUTthesis}[2023-02-01 v1.1 Standard LaTeX document class]
\ProcessOptions\relax
%% 读取基类
\LoadClass[a4paper,12pt]{article}
% \LoadClass[UTF8,a4paper,twoside,zihao=-4]{ctexbook}
\RequirePackage{ifxetex}
\RequireXeTeX
\ifxetex\else
\ClassError{glutthesis}{You must use the `xelatex' driver\MessageBreak Please choose `xelatex'}{%
Just choose `xelatex', no `pdflatex' or `latex' and so on.}
\fi

%---------------------------------------------------%
% 相关宏包 %
%---------------------------------------------------%
% 支持中文的 ctex 宏包
\RequirePackage{ctex}
% 页面布局
\RequirePackage{geometry}
% 数学宏包
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{bm}
\RequirePackage{newtxmath} %公式字体类似 Times New Roman
% 自定义关键词
\RequirePackage{pgfkeys}
% 设置目录
\RequirePackage{titletoc}
% 设置字体
\RequirePackage{fontenc}
% 设置颜色
\RequirePackage{xcolor}
% 设置页眉和页脚
\RequirePackage{fancyhdr}
% 设置标题格式
\RequirePackage{titlesec}
\newcommand{\sectionbreak}{\clearpage}
% 超链接 hyperref 的设置
\RequirePackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=black,citecolor=black}
% 插入图片
\RequirePackage{graphicx}
% 表格
\RequirePackage{array}
% 长表格
\RequirePackage{longtable}
% booktabs 提供了\toprule 等命令，三线表用
\RequirePackage{booktabs}
% multirow 支持在表格中跨行
\RequirePackage{multirow}
% 调整间隔, 让表格更好看些
\RequirePackage{bigstrut}
%在跨行表格中输入定界符
\RequirePackage{bigdelim}
% 保护脆落命令
\RequirePackage{cprotect}
% 设置代码环境
\RequirePackage{listings}
% 首行缩进
\RequirePackage{indentfirst}
% jing: ccaption宏包不能出现在 caption 宏包之后
% 设置浮动体的标题
\RequirePackage[justification=centering]{caption}
\RequirePackage[justification=centering]{subcaption}
% 定制列表环境，得到“第一行首行缩进两个字符，第二行顶格”的格式
\RequirePackage{enumitem}
\setenumerate{fullwidth,itemindent=\parindent,listparindent=\parindent,itemsep=0ex,partopsep=0pt,parsep=0ex}
% 提供\AtBeginEnvironment以方便全局调整一些结构的设置
\RequirePackage{etoolbox}
% 枚举
\RequirePackage{enumitem}
% lastpage
\RequirePackage{lastpage}

% 参考文献格式 GB/T7714-2015
% 来自https://github.com/hushidong/biblatex-gb7714-2015
\RequirePackage[backend=biber,gbpub=false, style=gb7714-2015]{biblatex}
% 使用系统字体
\usepackage{fontspec}
% 化学方程式
\RequirePackage[version=4]{mhchem} %化学方程式 
\RequirePackage{chemfig} %有机化学方程式 


%---------------------------------------------------%
% 页面布局 %
%---------------------------------------------------%
\geometry{top=3cm,bottom=2.5cm,left=3cm,right=2cm}
% 给页眉留足空间（否则会有 \headheight is too small的warning）
\setlength{\headheight}{24.5pt}
% “磅”是衡量印刷字体大小的单位,约等于七十二分之一英寸。
% 而 1英寸＝25.4毫米，则1磅＝25.4/72≈0.353毫米。
% 磅和 LaTeX的 pt- points (大约 1/72 inch) 是一致的。
\renewcommand*{\baselinestretch}{1.0}   % 几倍行间距
\setlength{\baselineskip}{20pt}         % 基准行间距
%\setlength{\bibsep}{0.5ex} % 参考文献条目间距

%---------------------------------------------------%
% 设置字体 %
%---------------------------------------------------%
\setmainfont{Times New Roman}
\setmonofont{Courier New}
\setsansfont{Arial}
\renewcommand\normalsize{%
	\@setfontsize\normalsize{12.05}{14.45}%
	\abovedisplayskip 12\p@ \@plus3\p@ \@minus7\p@
	\abovedisplayshortskip \z@ \@plus3\p@
    \belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@
}

% 设置宋体、黑体、楷体，以支持伪加粗，伪斜体
\setCJKmainfont{SimSun}[AutoFakeBold=2.5,AutoFakeSlant]
\setCJKsansfont{SimHei}[AutoFakeBold,AutoFakeSlant]
\setCJKmonofont{KaiTi}[AutoFakeBold,AutoFakeSlant]
%\let\kaishu\relax
%\newCJKfontfamily\kaishu{KaiTi}[AutoFakeBold,AutoFakeSlant]

% 实现华文行楷
\setCJKfamilyfont{hwxk}{STXingkai}[AutoFakeBold=2.5,AutoFakeSlant=0.2]
\newcommand{\huawenxingkai}{\CJKfamily{hwxk}}

% 实现宋体英文
\newfontfamily\mysong{SimSun}[AutoFakeBold=2.5,AutoFakeSlant=0.2]
% 实现楷体GB_2312
%\setCJKfamilyfont{kaitigb}{KaiTi_GB2312.ttf}
%\newcommand{\kgb}{\CJKfamily{kaitigb}}

% 设置参考文献的字体格式
\renewcommand{\bibfont}{\zihao{-4} }

% 每行缩进两个汉字
\setlength{\parindent}{2em}

%  中文标题名称
\newcommand*{\glut@cap@contentsname}{目录}
\newcommand*{\glut@cap@listfigurename}{插图清单}
\newcommand*{\glut@cap@listtablename}{附表清单}
\newcommand*{\glut@cap@refname}{参考文献}
\newcommand*{\glut@cap@indexname}{索引}
\newcommand*{\glut@cap@figurename}{图}
\newcommand*{\glut@cap@tablename}{表}
\newcommand*{\glut@cap@appendixname}{附录}
\newcommand*{\glut@cap@abstractname}{摘要}
\newcommand*{\glut@cap@keywordsname}{关键词}
\renewcommand{\lstlistingname}{代码}
%---------------------------------------------------%
%控制引用格式，选用上标引用
% \newcommand\supercite[2][]{%
%    \textsuperscript{\cite[#1]{#2}}}

%---------------------------------------------------%
% 定义命令和定义相应的宏。
%---------------------------------------------------%
% 预定义全局使用的文字，如姓名、专业等信息，在content/info.tex中定义

% 以下定义封面相关命令和相应的宏，以titleen为例，\titleen为实际在源码中使用的命令，一旦执行就会因为\gdef\@titleen{#1}使得其产生一个\@titleen的宏，而该宏能够在后续的环境中使用。
\newcommand*{\titlestyle}[1]{\gdef\@titlestyle{#1}}           % 题目类型
\newcommand*{\titlecn}[1]{\gdef\@titlecn{#1}}                 % 中文标题
\newcommand*{\titlecntwo}[1]{\gdef\@titlecntwo{#1}}           % 中文标题第二行
\newcommand*{\titleen}[1]{\gdef\@titleen{#1}}                 % 英文标题
\newcommand*{\major}[1]{\gdef\@major{#1}}                     % 专业
\newcommand*{\class}[1]{\gdef\@class{#1}}                     % 班级
\newcommand*{\department}[1]{\gdef\@department{#1}}           % 学院
\newcommand*{\supervisor}[1]{\gdef\@supervisor{#1}}           % 导师
\newcommand*{\authoren}[1]{\gdef\@authoren{#1}}               % 作者英文名
\newcommand*{\supervisoren}[1]{\gdef\@supervisoren{#1}}       % 导师英文名
\newcommand*{\subsupervisor}[1]{\gdef\@subsupervisor{#1}}     % 第二导师
\newcommand*{\studentid}[1]{\gdef\@studentid{#1}}             % 学号
\newcommand*{\donedata}[1]{\gdef\@donedata{#1}}               % 完成日期


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. 封面
% -----------------------------------------------%
% 定义 \makecoverpage 实现封面的重新绘制
\newcommand{\makecoverpage} {
    \thispagestyle{empty}
    \vspace*{-0.5cm}
    {\bfseries  \zihao{-4}%
			\begin{flushleft}
				\hfill\makebox[18mm][s]{学号:}\underline{\makebox[30mm][c]{\@studentid}}
				\vskip0.3cm
				\hfill\makebox[18mm][s]{题目类型:}\underline{\makebox[30mm][c]{\@titlestyle}}
				\vskip0.3cm
				\hfill\makebox[35mm][s]{\scalebox{0.8}[1]{（设计、论文、报告）}}\makebox[19mm][c]{}\\
				\vskip0.2cm
			\end{flushleft}
	}
	\par
	\vspace*{1.2cm} 
	{\centerline{\bfseries\zihao{0}\huawenxingkai{桂\,林\,理\,工\,大\,学}}}\par
	\vspace*{1.0cm}\par
	{\centerline{\bfseries\zihao{-2}\mysong{GUILIN UNIVERSITY OF TECHNOLOGY}}}\par
	
	\vspace*{1.0cm}
	{\centerline{\bfseries\zihao{0}\scalebox{0.8}[1]{\mysong{本\,科\,毕\,业\,设\,计\,(\,论\,文\,)}}}}
    \par
   % \begin{center}
   %     \includegraphics[height=6cm]{images/title.pdf}
   % \end{center}
    \vspace{1cm}

    {\bfseries\zihao{-3} 
		\begin{center}
		\makebox[12mm][s]{\bfseries 题目:} \underline{\makebox[100mm][c]{\bfseries \@titlecn}}\\
		    \vskip0.2cm
		\makebox[12mm][s]{\bfseries } \underline{\makebox[100mm][c]{\bfseries \@titlecntwo}}\\  
				\vskip2.5cm
		\makebox[30mm][s]{\bfseries 学院:} \underline{\makebox[60mm][c]{\bfseries \@department}}\\
				\vskip0.5cm
		\makebox[30mm][s]{\bfseries 专业（方向）:} \underline{\makebox[60mm][c]{\bfseries \@major}}\\
				\vskip0.5cm
		\makebox[30mm][s]{\bfseries 班级:} \underline{\makebox[60mm][c]{\bfseries \@class}} \\
				\vskip0.5cm
		\makebox[30mm][s]{\bfseries 学生:} \underline{\makebox[60mm][c]{\bfseries \@author}}\\
				\vskip0.5cm
		\makebox[30mm][s]{\bfseries 指导教师:} \underline{\makebox[60mm][c]{\bfseries \@supervisor}}\\
				\vskip1.6cm
                \@donedata
		\end{center}
	}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. 原创性声明和学位论文版权使用授权书 
% -----------------------------------------------%
% 单独设一页，排在扉页后
% 按模板，标题二号，内容四号，这里学校没有特别强调字体字号，
% 应该有一定自由度。

\newcommand{\announcement}{
   
    \linespread{1.5} % 这部分的行间距跟正文不一样
	\vspace*{1cm}
    {\centering \zihao{2}  \bfseries \makebox[56mm][s]{桂林理工大学}\par
	 毕业设计（论文）独创性声明 \par}

    \vspace{0.5cm}

    \setlength{\parindent}{2em} % 首行缩进
    {
        \zihao{-4}
本人声明所呈交的设计（论文）是我个人在指导教师指导下进行的研究工作及取得的研究成果。尽我所知，除了设计（论文）中特别加以标注和致谢的地方外，设计（论文）中不包含其他人或集体已经发表或撰写的研究成果，也不包含为获得桂林理工大学或其它教育机构的学位或证书而使用过的材料。对设计（论文）的研究成果做出贡献的个人和集体，均已作了明确的标明。本人完全意识到本声明的法律后果由本人承担。

        \vspace{0.5cm}
\begin{center}
设计（论文）作者签名：\qquad \qquad \qquad 日期：\qquad 年 \qquad 月 \qquad 日	
\end{center}
        
    }

    \vspace{1cm}

    {\centering \zihao{2}  \bfseries \makebox[56mm][s]{桂林理工大学}\par
	 设计（论文）使用授权声明 \par}

    \vspace{0.5cm}

    {
        \zihao{-4} 
本设计（论文）作者完全了解学校有关保留、使用设计（论文）的规定，同意学校保留并向国家有关部门或机构送交设计（论文）的复印件和电子版，允许设计（论文）被查阅或借阅。本人授权桂林理工大学可以将本设计（论文）的全部或部分内容编入有关数据库进行检索，可以采用影印、缩印或扫描等复制手段保存和汇编本设计（论文）。

        \vspace{1cm}

\begin{center}
\makebox[46mm][s]{设计（论文）作者签名：}\qquad  \qquad \qquad 日期：\qquad 年 \qquad 月 \qquad 日\par
\makebox[46mm][s]{指导老师签名：}\qquad \qquad \qquad 日期：\qquad 年 \qquad 月 \qquad 日	
\end{center}
    }
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. 中文摘要 
% -----------------------------------------------%

\newcommand*{\keywordscn}[1]{\gdef\@keywordscn{#1}} % 定义中文关键词
\newenvironment{abstractcn} { % 定义中文摘要环境

    \linespread{1.5}
	\vspace*{-0.5cm}
    \begin{center} \zihao{3} \bfseries 摘\quad 要 \end{center}
    \vspace{0.5cm}
    \zihao{-4} %\noindent % 内容为小四宋体
}
% 显示中文关键词
{
    \begin{flushleft}
        \noindent {\zihao{-4} \bfseries  关键词：} {\zihao{-4}  \@keywordscn} \par % 关键词为四号黑体，内容为小四宋体
    \end{flushleft}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4. 英文摘要 %
% -----------------------------------------------%

% 定义英文摘要，字体全部为times new roman
\newcommand*{\keywordsen}[1]{\gdef\@keywordsen{#1}}    % 英文关键词
\newenvironment{abstracten} { % 新建英文摘要环境
    \linespread{1.5}
	\vspace*{-0.5cm}
    \begin{center} \zihao{-3} \bfseries \@titleen \end{center}
    \vspace{-0.8cm}
    \begin{center} \zihao{-4} Student:~\@authoren \qquad \quad Teacher:~\@supervisoren \end{center}
    
    \vspace{0.5cm}

    \noindent {\zihao{-4} \bfseries Abstract:~} % 摘要俩字为四号黑体
    \zihao{-4} \noindent % 内容为四号宋体
}
{
    \begin{flushleft}
        \noindent {\zihao{-4} \bfseries Key words:~} {\zihao{-4} \@keywordsen} \par % 
    \end{flushleft}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5. 目录 %
% -----------------------------------------------%
% 目录应列出论文的大标题、一级和二级节标题，逐项标明页码（具体见样例10）。各级标题应简明扼要、点出各部分主要内容。学位论文的页码编排为：正文和后置部分用阿拉伯数字编连续码，前置部分用罗马数字单独编连续码（封面除外）。
% “目录”两字居中（三号黑体），下空两行为章、节、小节及其开始页码（靠右顶格）。章、节、小节分别以阶梯式排列：1（第1章）、1.1、1.1.1依次标出。章、节、小节的题名与页码之间用“．．．．．．”连接。除“目录”两字外，其余字体均为小四号宋体。

\setcounter{secnumdepth}{3}
% 定义作为标题的格式
% \titleformat{command}[shape]%定义标题类型和标题样式
%               {format}%定义标题格式
%               {label}%定义标题的标签，即标题的标号等
%               {sep}%定义标题和标号之间的水平距离
%               {before-code}%定义标题前的内容
%               [after-code]%定义标题后的内容

%\newcommand{\chaptername}{第{\thesection}章}
\titleformat{\section}{\zihao{3} \bfseries \centering}{\,\thesection\,}{1em}{}%\chaptername
\titleformat{\subsection}{\zihao{-3}  \bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\zihao{4}  \bfseries}{\thesubsubsection}{1em}{}

% 定义在目录中的格式
% \titlecontents{标题名}
%               [左间距]
%               {标题格式}
%               {标题标志}
%               {无序号标题}
%               {指引线与页码}
%               [下间距]

\titlecontents{section}
    [0.0em]
    {\zihao{-4}\bfseries\songti}									
    {\,\thecontentslabel\, \hspace{0.5em}}%第几章
    {}
    {~\titlerule*[0.35pc]{\tiny  \textbullet}~\contentspage}
              
\titlecontents{subsection}
    [3.6em]
    {\zihao{-4} \songti} % note that 3.8 = 1.5 + 2.3
    {\contentslabel{2.2em}}
    {\hspace*{-3.2em}}
    {~\titlerule*[0.35pc]{\tiny  \textbullet}~\contentspage}

\titlecontents{subsubsection}
    [5.8em]
    {\zihao{-4} \songti} % note that 3.8 = 1.5 + 2.3
    {\contentslabel{3.0em}}
    {\hspace*{-3.2em}}
    {~\titlerule*[0.35pc]{\tiny \textbullet}~\contentspage}
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 6. 论文正文 
% -----------------------------------------------%
% 论文正文是主体，主体部分应从另页右页开始，每一章应另起页。一般由序号标题、文字叙述、图、表格和公式等五个部分构成。
% 写作形式可因研究内容的性质不同而变化，一般可包括绪论（或综述）、理论分析、计算方法、实验装置和测试方法、实验结果分析和讨论、研究成果、结论及意义等。
% 该部分由用户编写

% 图表等浮动环境设置 %
% ----------------------------------------------------------%
% 图、表、公式：文中的图、表、附注、公式一律采用阿拉伯数字分章（或连续）编号，如：图2.5，表3.2，公式（5-1）等。
% 图序及图名居中置于图的下方，如果图中含有几个不同部分，应将分图号标注在分图的左上角，并在图解下列出各部分内容。图中的术语、符号、单位等应与正文表述所用一致。
% 表序及表名置于表的上方，表中参数应标明量和单位的符号。表的编排应采用国际通用的三线表。续表均应重复表头，如表需转页接排，随后各页应重复表的编号，其后跟表题（可省略）和“续”置于表上方。
% 图序及图名、表序及表名采用五号楷体字。若图或表中有附注，采用英文小写字母顺序编号，附注写在图或表的下方。公式的编号用括号括起写在右边行末，其间不加虚线。
% 图、表、公式等与正文之间要有一定的行间距。
% \RequirePackage{caption}
% 默认情况下, \LaTeX{} 要求每页的文字至少占据 20%,否则该页就只单独放置一个浮动环境,
% 而这通常不是我们想要的, 我们将这个要求降低到 5%.
\renewcommand*{\textfraction}{0.05}
% 有时如果多个浮动环境连续放在一起, \LaTeX{}
% 会将它们分在几个不同页，即使它们可在同一页放
% 得下. 我们可以通过修改 |\topfraction| 和 |\bottomfraction| 分别设置顶端和底端的浮
% 动环境的最大比例.
 \renewcommand*{\topfraction}{0.9}
 \renewcommand*{\bottomfraction}{0.8}
% 有时\LaTeX{}会把一个浮动环境单独放在一页,
% 我们要求这个环境至少要占据 85% 才能单独放在一页.
% 注意:  |\floatpagefraction| 的数值必须小于 |\topfraction|.
 \renewcommand*{\floatpagefraction}{0.85}
% 关于图片 graphicx
 % 如果图片没有指定后缀, 依次按下列顺序搜索
\DeclareGraphicsExtensions{.pdf,.eps,.jpg,.png,.tif}
 % 设置图表搜索路径, 可以给图表文件夹取如下名字
\graphicspath{{figures/}{figure/}{pictures/}%
  {picture/}{pic/}{pics/}{image/}{images/}}


% 插图格式
% ------------------------------------------------------%
% \captionsetup[subfigure]{labelfont=normalfont,textfont=normalfont,singlelinecheck=off,justification=raggedright}
% singlelinecheck=off 表示即使caption只有一行， justification 也生效
% justification=raggedright 使子图 caption 靠左对齐，而caption的上下位置由\caption标签的位置决定，以此实现了学校要求的子图标签在左上角显示。
% 这部分功能由 subcaption 这个包实现，而这个包不能和 subfigure 同时使用，所以子图的写法也与 subfigure 包的不大一样。
\captionsetup[subfigure]{singlelinecheck=off,justification=raggedright}

% 图下方描述的黑体
%\DeclareCaptionFont{hei}{\heiti}
%\DeclareCaptionFont{five}{\zihao{5}}
\renewcommand{\thefigure}{\arabic{section}-\arabic{figure}}
% \renewcommand{\thefigure} {\thesection-\arabic{figure}}

\captionsetup[figure]{
	format=plain,      % 标题从第二行开始是否缩进，plain无缩进，hang有缩进
	labelsep=quad,     % 分隔符是一个空格
	%font={hei,five},
	position=bottom    % position=bottom, 不代表标题放在下面, 标题仍放在你放\caption的位置.
}
% 表格格式
% ------------------------------------------------------%
\renewcommand{\thetable}{\arabic{section}-\arabic{table}}
% \renewcommand {\thetable} {\thesection-\arabic{table}}
\captionsetup[table]{
	format=plain,      % 标题从第二行开始是否缩进，plain无缩进，hang有缩进
	labelsep=quad,     % 分隔符是一个空格
	%font={hei,five},   % 表的字体, 宋体小四
	position=top
}

% 表格 %
% ------------------------------------------------------%
% 修改tabular 环境, 设置表格中的行间距为正文行间距.
\let\glut@oldtabular\tabular
\let\glut@endoldtabular\endtabular
\renewenvironment{tabular} {
    \bgroup
	\renewcommand{\arraystretch}{1.38}
    \glut@oldtabular
} {
    \glut@endoldtabular\egroup
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 设置宏包 listings/lstlistings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
             
%\RequirePackage{listings}

\lstset{tabsize=4,  %
  % frame=tcolorbox,  % 把代码用框圈起来
  % commentstyle=\color{red!50!green!50!blue!50},  % 浅灰色的注释
  % rulesepcolor=\color{red!20!green!20!blue!20},  % 代码块边框为淡青色
  keywordstyle=\color{black}\bfseries,  %代码关键字的颜色为蓝色，粗体
  showstringspaces=false,      % 不显示代码字符串中间的空格标记
  stringstyle=\ttfamily,       % 代码字符串的特殊格式
  keepspaces=true,             %
  breakindent=22pt,            %
  numbers=left,                % 左侧显示行号
  stepnumber=1,                %
  numberstyle=\zihao{-5},      % 行号字体
  basicstyle={\zihao{-4}\ttfamily},  %
  showspaces=false,            %
  flexiblecolumns=true,        %
  breaklines=true,             % 对过长的代码自动换行
  breakautoindent=true,        %
  breakindent=4em,             %
  aboveskip=1em,               % 代码块边框
  fontadjust,
  captionpos=t,
 framextopmargin=2pt,framexbottommargin=2pt,abovecaptionskip=-9pt,belowcaptionskip=9pt,
  xleftmargin=2em,xrightmargin=0em,  % 设定listing左右的空白
  texcl=true,
  % 设定中文冲突，断行，列模式，数学环境输入，listing数字的样式
  extendedchars=false,columns=flexible,mathescape=true
  numbersep=-1em
}

% 数学环境, 定理等设置 %
% ------------------------------------------------------%
\newtheorem{definition}{\glut@cap@definition}
\newtheorem{theorem}{\glut@cap@theorem}
\newtheorem{lemma}{\glut@cap@lemma}
\newtheorem{corollary}{\glut@cap@corollary}
\newtheorem{assumption}{\glut@cap@assumption}
\newtheorem{conjecture}{\glut@cap@conjecture}
\newtheorem{axiom}{\glut@cap@axiom}
\newtheorem{principle}{\glut@cap@principle}
\newtheorem{problem}{\glut@cap@problem}
\newtheorem{example}{\glut@cap@example}
\newtheorem{proof}{\glut@cap@proof}
\newtheorem{solution}{\glut@cap@solution}

% 数学定理相关的常量
\newcommand*{\glut@cap@definition}{定义}
\newcommand*{\glut@cap@theorem}{定理}
\newcommand*{\glut@cap@lemma}{引理}
\newcommand*{\glut@cap@corollary}{推论}
\newcommand*{\glut@cap@assumption}{假设}
\newcommand*{\glut@cap@conjecture}{猜想}
\newcommand*{\glut@cap@axiom}{公理}
\newcommand*{\glut@cap@principle}{定律}
\newcommand*{\glut@cap@problem}{问题}
\newcommand*{\glut@cap@example}{例}
\newcommand*{\glut@cap@proof}{证明}
\newcommand*{\glut@cap@solution}{解}

% 编号分章节。如需要连续编号，注释\makeatletter下面对应内容即可。
\renewcommand{\theequation}{\arabic{section}-\arabic{equation}}
\makeatletter
\@addtoreset{figure}{section}
\@addtoreset{table}{section}
\@addtoreset{equation}{section}
\makeatother
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 7. 参考文献
% -----------------------------------------------%
% 文后参考文献只列出作者直接阅读过、在正文中被引用过的文献资料，务必实事求是。参考文献一律列在正文的末尾，不得放在各章之后。人文社科类学位论文中不宜用文后参考文献列出的注释可采用页下注，包括对学位论文中某些关键词句、论点的详细说明。在引用别人的科研成果时，应在引用处加以说明，遵循学术道德规范，严禁论文抄袭、剽窃等学术不端行为。
% 参考文献采用顺序编码制组织，即按中文引用的顺序将参考文献附于文末。作者姓名写到第三位，余者写“，等”或“，et al.”。左起按顺序依次列出参考文献，将序号置于方括号内(如[1])，用小四号宋体字，所有符号均用半角编排。
% 目前采用bgt7714-2005标准进行文献索引，详见gbt7714-2005.bst.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 8. 致谢 %
% -----------------------------------------------%
% 作者对给予指导、各类资助和协助完成研究工作以及提供各种对论文工作有利条件的单位及个人表示感谢。
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 9. 附录（必要时）%
% -----------------------------------------------%
% 附录中主要列入正文内不便列出的过分冗长的公式推导，供查读方便所需的辅助性数学工具或表格、重复性数据图表、计算程序及说明等。附录依次为附录1，附录2……等，“附录X”三字居中。附录中的图表公式另编排序号，与正文分开。

\endinput
