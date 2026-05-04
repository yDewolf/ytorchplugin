## Built with [nodeblocks-server](https://github.com/yDewolf/nodeblocks-server)

## Como rodar/desenvolver:
Depois de clonar o repositório:
````
pip install -e ./lib/nodeserver
pip install -e .
````

Ideias principais:
- Interface web que permite você pegar vários blocos de construção (módulos de NN do pytorch, ex: Conv2D) e conectar os outputs deles para criar um modelo
- O website deve fornecer avisos sobre shapes, assim como mostrar o shape previsto de cada output
- A ideia é salvar os outputs e inputs em um cache para que eu possa fazer um modo "stop motion", que passa por cada camada do modelo em forma de uma animaçãozinha
- Os inputs de cada módulo do modelo devem poder ser qualquer output de outro módulo, ou o input inicial
- Deve existir a possibilidade de "agendar" reshapes, permutes e views, para corrigir os shapes dos outputs

Acho que é basicamente isso (vou levar 3 anos para conseguir terminar)