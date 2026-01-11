Wrong-format references (converted to citation commands) — 2025-12-22

Source locations (line numbers from `appendix.tex`):
- [x] L231 NumPy: replace “Harris et al. (2020)” → `\citep{harris2020numpy}`
- [x] L232 pandas: replace “McKinney (2010)” → `\citep{mckinney2010pandas}`
- [x] L233 SciPy: replace “Virtanen et al. (2020)” → `\citep{virtanen2020scipy}`
- [x] L236 scikit-learn: replace “Pedregosa et al. (2011)” → `\citep{pedregosa2011sklearn}`
- [x] L237 imbalanced-learn: replace “Lemaître et al. (2017)” → `\citep{lemaitre2017imblearn}`
- [x] L240 Matplotlib: replace “Hunter (2007)” → `\citep{hunter2007matplotlib}`
- [x] L241 seaborn: replace “Waskom (2021)” → `\citep{waskom2021seaborn}`
- [x] L244 NetworkX: replace “Hagberg et al. (2008)” → `\citep{hagberg2008networkx}`
- [x] L245 Neo4j: replace “Robinson et al. (2013)” → `\citep{robinson2013neo4j}`
- [x] L248 Pydantic: replace “Colvin et al. (2024)” → `\citep{pydantic2024}`
- [x] L251 Jupyter: replace “Kluyver et al. (2016)” → `\citep{kluyver2016jupyter}`

 Key references (with URLs & BibTeX keys):
- `harris2020numpy` — https://doi.org/10.1038/s41586-020-2649-2
- `mckinney2010pandas` — https://doi.org/10.25080/Majora-92bf1922-00a
- `virtanen2020scipy` — https://doi.org/10.1038/s41592-019-0686-2
- `pedregosa2011sklearn` — https://jmlr.org/papers/v12/pedregosa11a.html
- `lemaitre2017imblearn` — http://jmlr.org/papers/v18/16-365.html
- `hunter2007matplotlib` — https://doi.org/10.1109/MCSE.2007.55
- `waskom2021seaborn` — https://doi.org/10.21105/joss.03021
- `hagberg2008networkx` — Proc. SciPy 2008 (11–15) — https://conference.scipy.org/proceedings/SciPy2008/paper_2/
- `robinson2013neo4j` — Graph Databases (O’Reilly), ISBN 978-1-4493-5626-2
- `pydantic2024` — https://github.com/pydantic/pydantic (Zenodo DOI:10.5281/zenodo.8421425)
- `kluyver2016jupyter` — https://jupyter.org

Additional author–year instances to convert to `\citep{...}` (from full-thesis scan):
- [x] CaLM benchmark — `chen2024calm` — https://arxiv.org/abs/2405.00622
- [x] Corr2Cause — `jin2023corr2cause` — https://arxiv.org/abs/2306.05836
- [x] Causal Parrots — `zecevic2023causalparrots` — https://arxiv.org/abs/2308.13067
- [x] Pipeline Algebra / Systems taxonomy — `reinholtz2025pipelinealgebra` — URL: (not publicly available yet; journal “Systems”, 2025)
- [x] World Dynamics — `forrester1971worlddynamics` — https://archive.org/details/worlddynamics00jayw
- [x] Limits to Growth — `meadows1972limits` — https://archive.org/details/limitstogrowth00mead
- [x] Lotka predator–prey — `lotka1925elements` — https://archive.org/details/elementsofphysic00lotkrich
- [x] Volterra predator–prey — `volterra1926fluctuations` — https://doi.org/10.1002/andp.19263892006
- [x] Kermack & McKendrick SIR — `kermack1927sir` — https://doi.org/10.1098/rspa.1927.0118
- [x] IPCC AR6 WG1 — `ipcc2021ar6wg1` — https://www.ipcc.ch/report/ar6/wg1/
- [x] Sterman beer-game reference — `sterman1989misperceptions` — https://doi.org/10.1287/mnsc.35.3.321
- [x] Kojima CoT — `kojima2022large` — https://arxiv.org/abs/2205.11916
- [x] Shimonovich Bradford Hill — `shimonovich2020bradford` — https://doi.org/10.1007/s10654-020-00673-4


@article{harris2020array,
  title={Array programming with NumPy},
  author={Harris, Charles R and Millman, K Jarrod and Van Der Walt, St{\'e}fan J and Gommers, Ralf and Virtanen, Pauli and Cournapeau, David and Wieser, Eric and Taylor, Julian and Berg, Sebastian and Smith, Nathaniel J and others},
  journal={nature},
  volume={585},
  number={7825},
  pages={357--362},
  year={2020},
  publisher={Nature Publishing Group UK London}
}


@article{mckinney2010data,
  title={Data structures for statistical computing in Python.},
  author={McKinney, Wes and others},
  journal={scipy},
  volume={445},
  number={1},
  pages={51--56},
  year={2010}
}

@article{virtanen2020scipy,
  title={SciPy 1.0: fundamental algorithms for scientific computing in Python},
  author={Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E and Haberland, Matt and Reddy, Tyler and Cournapeau, David and Burovski, Evgeni and Peterson, Pearu and Weckesser, Warren and Bright, Jonathan and others},
  journal={Nature methods},
  volume={17},
  number={3},
  pages={261--272},
  year={2020},
  publisher={Nature Publishing Group US New York}
}

@article{pedregosa2011scikit,
  title={Scikit-learn: Machine learning in Python},
  author={Pedregosa, Fabian and Varoquaux, Ga{\"e}l and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and Dubourg, Vincent and others},
  journal={the Journal of machine Learning research},
  volume={12},
  pages={2825--2830},
  year={2011},
  publisher={JMLR. org}
}


@article{JMLR:v18:16-365,
  author  = {Guillaume  Lema{{\^i}}tre and Fernando Nogueira and Christos K. Aridas},
  title   = {Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2017},
  volume  = {18},
  number  = {17},
  pages   = {1--5},
  url     = {http://jmlr.org/papers/v18/16-365.html}
}

@ARTICLE{4160265,
  author={Hunter, John D.},
  journal={Computing in Science & Engineering}, 
  title={Matplotlib: A 2D Graphics Environment}, 
  year={2007},
  volume={9},
  number={3},
  pages={90-95},
  keywords={Graphics;Interpolation;Equations;Graphical user interfaces;Packaging;Image generation;User interfaces;Operating systems;Computer languages;Programming profession;Python;scripting languages;application development;scientific programming},
  doi={10.1109/MCSE.2007.55}}

  @article{Waskom2021, doi = {10.21105/joss.03021}, url = {https://doi.org/10.21105/joss.03021}, year = {2021}, publisher = {The Open Journal}, volume = {6}, number = {60}, pages = {3021}, author = {Waskom, Michael L.}, title = {seaborn: statistical data visualization}, journal = {Journal of Open Source Software} }


  @software{Colvin_Pydantic_Validation_2025,
author = {Colvin, Samuel and Jolibois, Eric and Ramezani, Hasan and Garcia Badaracco, Adrian and Dorsey, Terrence and Montague, David and Matveenko, Serge and Trylesinski, Marcelo and Runkle, Sydney and Hewitt, David and Hall, Alex and Plot, Victorien},
license = {MIT},
month = oct,
title = {{Pydantic Validation}},
url = {https://github.com/pydantic/pydantic},
version = {v2.13.0a0+dev},
year = {2025}
}

@article{robinson2013graph,
  title={Graph Databases O’Reilly Media},
  author={Robinson, Ian and Webber, Jim and Eifrem, Emil},
  journal={Cambridge, USA},
  year={2013}
}


@InProceedings{SciPyProceedings_11,
  author =       {Aric A. Hagberg and Daniel A. Schult and Pieter J. Swart},
  title =        {Exploring Network Structure, Dynamics, and Function using NetworkX},
  booktitle =   {Proceedings of the 7th Python in Science Conference},
  pages =     {11 - 15},
  address = {Pasadena, CA USA},
  year =      {2008},
  editor =    {Ga\"el Varoquaux and Travis Vaught and Jarrod Millman},
}


@incollection{kluyver2016jupyter,
  title={Jupyter Notebooks--a publishing format for reproducible computational workflows},
  author={Kluyver, Thomas and Ragan-Kelley, Benjamin and P{\'e}rez, Fernando and Granger, Brian and Bussonnier, Matthias and Frederic, Jonathan and Kelley, Kyle and Hamrick, Jessica and Grout, Jason and Corlay, Sylvain and others},
  booktitle={Positioning and power in academic publishing: Players, agents and agendas},
  pages={87--90},
  year={2016},
  publisher={IOS press}
}


@misc{chen2024causalevaluationlanguagemodels,
      title={Causal Evaluation of Language Models}, 
      author={Sirui Chen and Bo Peng and Meiqi Chen and Ruiqi Wang and Mengying Xu and Xingyu Zeng and Rui Zhao and Shengjie Zhao and Yu Qiao and Chaochao Lu},
      year={2024},
      eprint={2405.00622},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.00622}, 
}


@misc{jin2024largelanguagemodelsinfer,
      title={Can Large Language Models Infer Causation from Correlation?}, 
      author={Zhijing Jin and Jiarui Liu and Zhiheng Lyu and Spencer Poff and Mrinmaya Sachan and Rada Mihalcea and Mona Diab and Bernhard Schölkopf},
      year={2024},
      eprint={2306.05836},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2306.05836}, 
}


@misc{zečević2023causalparrotslargelanguage,
      title={Causal Parrots: Large Language Models May Talk Causality But Are Not Causal}, 
      author={Matej Zečević and Moritz Willig and Devendra Singh Dhami and Kristian Kersting},
      year={2023},
      eprint={2308.13067},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2308.13067}, 
}


@Article{systems13090784,
AUTHOR = {Reinholtz, Kirk and Shahroudi, Kamran Eftekhari and Lawrence, Svetlana},
TITLE = {LLM-Powered, Expert-Refined Causal Loop Diagramming via Pipeline Algebra},
JOURNAL = {Systems},
VOLUME = {13},
YEAR = {2025},
NUMBER = {9},
ARTICLE-NUMBER = {784},
URL = {https://www.mdpi.com/2079-8954/13/9/784},
ISSN = {2079-8954},
ABSTRACT = {Building a causal-loop diagram (CLD) is central to system-dynamics modeling but demands domain insight, the mastery of CLD notation, and the ability to juggle AI, mathematical, and execution tools. Pipeline Algebra (PA) reduces that burden by treating each step—LLM prompting, symbolic or numeric computation, algorithmic transforms, and cloud execution—as a typed, idempotent operator in one algebraic expression. Operators are intrinsically idempotent (implemented through memoization), so every intermediate result is re-used verbatim, yielding bit-level reproducibility even when individual components are stochastic. Unlike DAG (directed acyclic graph) frameworks such as Airflow or Snakemake, which force analysts to wire heterogeneous APIs together with glue code, PA’s compact notation lets them think in the problem space, rather than in workflow plumbing—echoing Iverson’s dictum that “notation is a tool of thought.” We demonstrated PA on a peer-reviewed study of novel-energy commercialization. Starting only from the article’s abstract, an AI-extracted problem statement, and an AI-assisted web search, PA produced an initial CLD. A senior system-dynamics practitioner identified two shortcomings: missing best-practice patterns and lingering dependence on the problem statement. A one-hour rewrite that embedded best-practice rules, used iterative prompting, and removed the problem statement yielded a diagram that conformed to accepted conventions and better captured the system. The results suggest that earlier gaps were implementation artifacts, not flaws in PA’s design; quantitative validation will be the subject of future work.},
DOI = {10.3390/systems13090784}
}

@article{boyd1972world,
  title={World dynamics: a note},
  author={Boyd, Robert},
  journal={Science},
  volume={177},
  number={4048},
  pages={516--519},
  year={1972},
  publisher={American Association for the Advancement of Science}
}

@article{dorgo2018automated,
  title={Automated analysis of the interactions between sustainable development goals extracted from models and texts of sustainability science},
  author={Dorgo, Gyula and Honti, Gergely and Abonyi, J{\'a}nos},
  journal={Chemical Engineering Transactions},
  volume={70},
  pages={781--786},
  year={2018}
}


@article{lotka1925predator,
  title={Predator-prey model},
  author={Lotka, Alfred James and Volterra, Vito},
  journal={Elements of Physical Biology},
  year={1925}
}


@incollection{bacaer2011lotka,
  title={Lotka, Volterra and the predator--prey system (1920--1926)},
  author={Baca{\"e}r, Nicolas},
  booktitle={A short history of mathematical population dynamics},
  pages={71--76},
  year={2011},
  publisher={Springer}
}


@article{10.1098/rspa.1927.0118,
    author = {Kermack, William Ogilvy and McKendrick, A. G.},
    title = {A contribution to the mathematical theory of epidemics},
    journal = {Proceedings of the Royal Society of London. Series A, Containing Papers of a Mathematical and Physical Character},
    volume = {115},
    number = {772},
    pages = {700-721},
    year = {1927},
    month = {08},
    abstract = {(1) One of the most striking features in the study of epidemics is the difficulty of finding a causal factor which appears to be adequate to account for the magnitude of the frequent epidemics of disease which visit almost every population. It was with a view to obtaining more insight regarding the effects of the various factors which govern the spread of contagious epidemics that the present investigation was undertaken. Reference may here be made to the work of Ross and Hudson (1915-17) in which the same problem is attacked. The problem is here carried to a further stage, and it is considered from a point of view which is in one sense more general. The problem may be summarised as follows: One (or more) infected person is introduced into a community of individuals, more or less susceptible to the disease in question. The disease spreads from the affected to the unaffected by contact infection. Each infected person runs through the course of his sickness, and finally is removed from the number of those who are sick, by recovery or by death. The chances of recovery or death vary from day to day during the course of his illness. The chances that the affected may convey infection to the unaffected are likewise dependent upon the stage of the sickness. As the epidemic spreads, the number of unaffected members of the community becomes reduced. Since the course of an epidemic is short compared with the life of an individual, the population may be considered as remaining constant, except in as far as it is modified by deaths due to the epidemic disease itself. In the course of time the epidemic may come to an end. One of the most important probems in epidemiology is to ascertain whether this termination occurs only when no susceptible individuals are left, or whether the interplay of the various factors of infectivity, recovery and mortality, may result in termination, whilst many susceptible individuals are still present in the unaffected population. It is difficult to treat this problem in its most general aspect. In the present communication discussion will be limited to the case in which all members of the community are initially equally susceptible to the disease, and it will be further assumed that complete immunity is conferred by a single infection.},
    issn = {0950-1207},
    doi = {10.1098/rspa.1927.0118},
    url = {https://doi.org/10.1098/rspa.1927.0118},
    eprint = {https://royalsocietypublishing.org/rspa/article-pdf/115/772/700/24858/rspa.1927.0118.pdf},
}

@article{huang2023ar5,
  title={From AR5 to AR6: Exploring research advancement in climate change based on scientific evidence from IPCC WGI reports},
  author={Huang, Tian-Yuan and Ding, Liangping and Yu, Yong-Qiang and Huang, Lei and Yang, Liying},
  journal={Scientometrics},
  volume={128},
  number={9},
  pages={5227--5245},
  year={2023},
  publisher={Springer}
}


@article{sterman1989modeling,
  title={Modeling managerial behavior: Misperceptions of feedback in a dynamic decision making experiment},
  author={Sterman, John D},
  journal={Management science},
  volume={35},
  number={3},
  pages={321--339},
  year={1989},
  publisher={INFORMS}
}


@misc{kojima2023largelanguagemodelszeroshot,
      title={Large Language Models are Zero-Shot Reasoners}, 
      author={Takeshi Kojima and Shixiang Shane Gu and Machel Reid and Yutaka Matsuo and Yusuke Iwasawa},
      year={2023},
      eprint={2205.11916},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2205.11916}, 
}

@article{shimonovich2024applying,
  title={Applying Bradford Hill to assessing causality in systematic reviews: A transparent approach using process tracing},
  author={Shimonovich, Michal and Thomson, Hilary and Pearce, Anna and Katikireddi, Srinivasa Vittal},
  journal={Research Synthesis Methods},
  volume={15},
  number={6},
  pages={826--838},
  year={2024},
  publisher={Wiley Online Library}
}

@misc{chen2024causalevaluationlanguagemodels,
      title={Causal Evaluation of Language Models}, 
      author={Sirui Chen and Bo Peng and Meiqi Chen and Ruiqi Wang and Mengying Xu and Xingyu Zeng and Rui Zhao and Shengjie Zhao and Yu Qiao and Chaochao Lu},
      year={2024},
      eprint={2405.00622},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.00622}, 
}









