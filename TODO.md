- [X] # Checar se dá mesmo para "montar" modelos em runtime

- [ ] Fazer um Backward (que tenha suporte para loss functions customizadas no futuro)
- [ ] Adicionar cálculos de shape estáticos (para evitar ficar criando tensores só para ver qual é o output shape do módulo) (no [ShapeCalculator](src/python/helpers/ShapeCalculator.py))

- [ ] mudar a função from_type_str para funcionar de uma forma mais decente (pra eu não ter que adicionar cada módulo do torch ali) [ModuleParts](src\python\data_types\ModuleParts.py)

Mais para frente:
- [ ] Interface de edição com base em nodes (provavelmente web)
- [ ] Visualização de cada forward step (exemplo, module_forward ou reshape)
- [ ] Gerar nn.Sequentials com base na estrutura do forward para evitar sincronização da CPU com a GPU.