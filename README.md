# Análise de rede Comida di Buteco JF

### Introdução

Este notebook analisa a rede viária de Juiz de Fora (MG) a partir das coordenadas dos 40 bares participantes do concurso "Comida di Buteco JF 2026". 

O objetivo é caracterizar a estrutura do grafo, identificar interseções movimentadas e avaliar a acessibilidade dos bares na rede viária.

### Objetivos específicos

- Construir o grafo viário da região central de JF (raio de 25 km)
- Calcular métricas básicas (ordem, tamanho, densidade, graus)
- Identificar nós com alta centralidade
- Verificar quantos bares estão próximos a interseções movimentadas (grau ≥ 6)
- Fornecer base para roteirização e análises futuras

### Importamos as bibliotecas


```python
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import folium
from folium.plugins import Fullscreen
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

```

### Verificamos as versões


```python
print(f"NetworkX version: {nx.__version__}")
print(f"OSMnx version: {ox.__version__}")
print(f"Folium version: {folium.__version__}")
```

    NetworkX version: 3.6.1
    OSMnx version: 2.1.0
    Folium version: 0.18.0


## Carregamos e inspecionamos os dados dos bares


**Carregamos a lista de bares**


```python
gdf = pd.read_csv("lista_bares.csv")
X = np.array(gdf[['latitude', 'longitude']])
```

**Inspecionamos os dados**


```python
print("=== Informações do dataset ===\n")
gdf.info()

print("\n=== Primeiras 5 coordenadas (lat, lon) ===\n")
print(X[:5])

print(f"\nTotal de bares carregados: {len(gdf)}")
print(f"Colunas disponíveis: {list(gdf.columns)}")
```

    === Informações do dataset ===
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 40 entries, 0 to 39
    Data columns (total 9 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Name           40 non-null     object 
     1   longitude      40 non-null     float64
     2   latitude       40 non-null     float64
     3   Endereço       40 non-null     object 
     4   Petisco        40 non-null     object 
     5   Contato        40 non-null     object 
     6   Instagram      40 non-null     object 
     7   Descrição      40 non-null     object 
     8   Funcionamento  40 non-null     object 
    dtypes: float64(2), object(7)
    memory usage: 2.9+ KB
    
    === Primeiras 5 coordenadas (lat, lon) ===
    
    [[-21.7819995 -43.2989666]
     [-21.7365987 -43.3609957]
     [-21.7586111 -43.3472222]
     [-21.766567  -43.3723106]
     [-21.7756168 -43.378489 ]]
    
    Total de bares carregados: 40
    Colunas disponíveis: ['Name', 'longitude', 'latitude', 'Endereço', 'Petisco', 'Contato', 'Instagram', 'Descrição', 'Funcionamento']


### Criamos o grafo viário


```python
# Calcular o centro geográfico dos bares
center = (X[:, 0].mean(), X[:, 1].mean())
print(f"Centro da região: latitude {center[0]:.6f}, longitude {center[1]:.6f}")

# Criar o grafo a partir do ponto central (raio de 25 km)
print("\nCriando grafo viário... Isso pode levar alguns segundos.")
G = ox.graph_from_point(center, dist=25000, network_type='drive')

# Adicionar atributos de velocidade e tempo de viagem
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

print(f"\n✅ Grafo criado com sucesso!")
```

    Centro da região: latitude -21.758952, longitude -43.359572
    
    Criando grafo viário... Isso pode levar alguns segundos.
    
    ✅ Grafo criado com sucesso!


### Estatísticas básicas do grafo


```python
print("=== Estatísticas básicas do grafo ===\n")
print(f"Tipo do grafo: {type(G).__name__}")
print(f"Direcionado? {G.is_directed()}")
print(f"Nós (ordem): {G.number_of_nodes():,}")
print(f"Arestas (tamanho): {G.number_of_edges():,}")
print(f"Ponderado? {len(list(G.edges(data=True)))} > 0")
```

    === Estatísticas básicas do grafo ===
    
    Tipo do grafo: MultiDiGraph
    Direcionado? True
    Nós (ordem): 13,086
    Arestas (tamanho): 30,461
    Ponderado? 30461 > 0


**Análise de conectividade**


```python
sccs = list(nx.strongly_connected_components(G))
print(f"\nNúmero de componentes fortemente conectados (SCCs): {len(sccs)}")
print(f"Tamanho do maior SCC: {len(max(sccs, key=len)):,}")
print(f"É fracamente conectado? {nx.is_weakly_connected(G)}")
```

    
    Número de componentes fortemente conectados (SCCs): 19
    Tamanho do maior SCC: 13,068
    É fracamente conectado? True


### Densidade do grafo


```python
# Cálculo da densidade para grafo direcionado
n = G.number_of_nodes()
m = G.number_of_edges()
densidade = m / (n * (n - 1))

print("=== Densidade do grafo ===\n")
print(f"Densidade: {densidade:.10f}")
```

    === Densidade do grafo ===
    
    Densidade: 0.0001778949


**Interpretação contextual**

Este valor extremamente baixo (≈ 0,00018) indica um grafo esparso.
Isso é típico de redes viárias urbanas, onde cada nó (interseção)
se conecta a apenas alguns vizinhos, formando uma malha eficiente
porém pouco densa — ao contrário de redes sociais ou de computadores.


### Análise de graus

**Grau de um nó específico (exemplo)**


```python
exemplo_no = 254452257
print(f"Grau do nó {exemplo_no}: {G.degree[exemplo_no]}")

```

    Grau do nó 254452257: 4


**Estatísticas gerais de grau**


```python
graus = dict(G.degree())
media_grau = np.mean(list(graus.values()))
max_grau = np.max(list(graus.values()))
mediana_grau = np.median(list(graus.values()))

print(f"\n📈 Estatísticas gerais:")
print(f"  - Média: {media_grau:.2f}")
print(f"  - Mediana: {mediana_grau:.1f}")
print(f"  - Máximo: {max_grau}")
```

    
    📈 Estatísticas gerais:
      - Média: 4.66
      - Mediana: 6.0
      - Máximo: 10


**Histograma de graus**


```python
hist = nx.degree_histogram(G)
degrees = list(range(len(hist)))

plt.figure(figsize=(12, 6))
plt.bar(degrees, hist, color='skyblue', edgecolor='black', alpha=0.7)

plt.xlabel('Grau do nó (número de conexões)', fontsize=12)
plt.ylabel('Frequência (número de nós)', fontsize=12)
plt.title('Distribuição de graus do grafo viário', fontsize=14, fontweight='bold')
plt.xticks(degrees)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Adicionar valores no topo das barras
for i, v in enumerate(hist):
    if v > 0:
        plt.text(i, v + 50, str(v), ha='center', fontsize=9)

plt.tight_layout()
plt.show()

print("\n Interpretação do histograma:")
print(f"- {hist[1]:,} nós têm grau 1 (ruas sem saída ou pontas da rede)")
print(f"- {hist[6]:,} nós têm grau 6 — possivelmente rotatórias ou largos")
print(f"- Apenas {hist[10] if len(hist) > 10 else 0} nós têm grau máximo (10)")

```


    
![png](output_28_0.png)
    


    
     Interpretação do histograma:
    - 18 nós têm grau 1 (ruas sem saída ou pontas da rede)
    - 6,395 nós têm grau 6 — possivelmente rotatórias ou largos
    - Apenas 3 nós têm grau máximo (10)


### Atributos de uma aresta (exemplo detalhado)


```python
# Pega a primeira aresta do grafo
first_edge = list(G.edges(data=True))[0]
u, v, data = first_edge

print("=== Detalhamento de uma aresta exemplo ===\n")
print(f"Nó origem: {u}")
print(f"Nó destino: {v}\n")
print("Atributos da aresta:")
for key, value in data.items():
    print(f"  - {key}: {value}")

print("\n Significado prático:")
print("- A Avenida Barão do Rio Branco é uma via secundária de mão única")
print("- 2 faixas, velocidade máxima de 60 km/h")
print("- O trecho de 138 metros é percorrido em aproximadamente 8,3 segundos")
```

    === Detalhamento de uma aresta exemplo ===
    
    Nó origem: 254452257
    Nó destino: 1348578073
    
    Atributos da aresta:
      - osmid: 279772695
      - highway: secondary
      - lanes: 2
      - maxspeed: 60
      - name: Avenida Barão do Rio Branco
      - oneway: True
      - reversed: False
      - length: 137.61977903639723
      - geometry: LINESTRING (-43.3484305 -21.7664571, -43.3484099 -21.7665324, -43.3481022 -21.7676566)
      - speed_kph: 60.0
      - travel_time: 8.257186742183835
    
     Significado prático:
    - A Avenida Barão do Rio Branco é uma via secundária de mão única
    - 2 faixas, velocidade máxima de 60 km/h
    - O trecho de 138 metros é percorrido em aproximadamente 8,3 segundos


### Identificação de nós de alta conectividade


```python
# Nós com grau elevado (≥ 6)
high_degree_nodes = [node for node in G.nodes() if G.degree(node) >= 6]
print(f"=== Nós de alta conectividade ===\n")
print(f"Nós com grau ≥ 6: {len(high_degree_nodes):,} de {n:,} nós totais")
print(f"Percentual: {len(high_degree_nodes)/n*100:.2f}%")

# Amostra dos 10 nós com maior grau
top_graus = sorted(graus.items(), key=lambda x: x[1], reverse=True)[:10]
print("\n🏆 Top 10 nós com maior grau:")
for i, (node, degree) in enumerate(top_graus, 1):
    print(f"  {i}. Nó {node}: grau {degree}")
```

    === Nós de alta conectividade ===
    
    Nós com grau ≥ 6: 7,055 de 13,086 nós totais
    Percentual: 53.91%
    
    🏆 Top 10 nós com maior grau:
      1. Nó 4411335432: grau 10
      2. Nó 2401994829: grau 10
      3. Nó 3149089186: grau 10
      4. Nó 256929276: grau 8
      5. Nó 256929297: grau 8
      6. Nó 256929447: grau 8
      7. Nó 256929560: grau 8
      8. Nó 256930209: grau 8
      9. Nó 1335008211: grau 8
      10. Nó 1339560833: grau 8


### Relação entre bares e nós do grafo


```python
from shapely.geometry import Point

print("=== Bares próximos a interseções movimentadas ===\n")

bar_locations = []
for idx, row in gdf.iterrows():  # CORRIGIDO: itterrows → iterrows
    point = Point(row['longitude'], row['latitude'])
    nearest_node = ox.nearest_nodes(G, row['longitude'], row['latitude'])
    bar_locations.append({
        'bar': row['Name'],
        'node': nearest_node,
        'node_degree': G.degree(nearest_node),
        'is_high_traffic': G.degree(nearest_node) >= 6
    })

# Bares localizados em interseções movimentadas
busy_intersection_bars = [b for b in bar_locations if b['is_high_traffic']]
print(f"Bares em interseções movimentadas (grau ≥ 6): {len(busy_intersection_bars)} de {len(gdf)}\n")

print("📋 Lista desses bares:")
for bar in busy_intersection_bars:
    print(f"  - {bar['bar']}: grau {bar['node_degree']}")

```

    === Bares próximos a interseções movimentadas ===
    
    Bares em interseções movimentadas (grau ≥ 6): 12 de 40
    
    📋 Lista desses bares:
      - ADEGA BAR: grau 6
      - BAR DIAS: grau 6
      - BAR DO BENE: grau 8
      - BAR DO BREJO: grau 6
      - BAR DO PASSARINHO: grau 6
      - BAR TORRESMO: grau 6
      - BUTECO DO PRINCIPE: grau 6
      - CASARAO BAR: grau 6
      - DON JUAN GASTRONOMIA E EVENTOS: grau 6
      - PETISQUEIRA: grau 6
      - RECANTO DOS MANACAS: grau 6
      - REZA FORTE: grau 6


### Medidas de centralidade

**Centralidade de grau (Degree Centrality)**


```python
print("=== Centralidade de grau ===\n")
deg_cent = nx.degree_centrality(G)
top5_deg = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:5]

print(" Top 5 nós por centralidade de grau:")
for node, score in top5_deg:
    print(f"  - Nó {node}: {score:.6f}")

# Histograma da centralidade de grau
plt.figure(figsize=(10, 5))
plt.hist(list(deg_cent.values()), bins=30, color='lightgreen', edgecolor='black')
plt.xlabel('Centralidade de grau')
plt.ylabel('Número de nós')
plt.title('Distribuição da centralidade de grau')
plt.grid(axis='y', alpha=0.3)
plt.show()
```

    === Centralidade de grau ===
    
     Top 5 nós por centralidade de grau:
      - Nó 4411335432: 0.000764
      - Nó 2401994829: 0.000764
      - Nó 3149089186: 0.000764
      - Nó 256929276: 0.000611
      - Nó 256929297: 0.000611



    
![png](output_37_1.png)
    


**Centralidade de intermediação (Betweenness Centrality)**


```python
print("=== Centralidade de intermediação (Betweenness) ===\n")
print("  Atenção: Este cálculo pode levar alguns minutos em grafos grandes.")
print("Usando amostragem (k=500) para acelerar o processamento...\n")

# Cálculo correto com amostragem para grafos grandes
bet_cent = nx.betweenness_centrality(G, k=500, normalized=True)
top5_bet = sorted(bet_cent.items(), key=lambda x: x[1], reverse=True)[:5]

print(" Top 5 nós por centralidade de intermediação:")
for node, score in top5_bet:
    print(f"  - Nó {node}: {score:.8f}")

print("\n Interpretação:")
print("Nós com alta betweenness atuam como 'pontes' ou 'gargalos' na rede.")
print("Eles são cruciais para o fluxo de tráfego e podem ser pontos de estrangulamento.")

```

    === Centralidade de intermediação (Betweenness) ===
    
      Atenção: Este cálculo pode levar alguns minutos em grafos grandes.
    Usando amostragem (k=500) para acelerar o processamento...
    
     Top 5 nós por centralidade de intermediação:
      - Nó 5987726256: 0.18320548
      - Nó 1355441834: 0.14208337
      - Nó 5377741516: 0.12789768
      - Nó 7828670046: 0.12725379
      - Nó 3017899851: 0.12476626
    
     Interpretação:
    Nós com alta betweenness atuam como 'pontes' ou 'gargalos' na rede.
    Eles são cruciais para o fluxo de tráfego e podem ser pontos de estrangulamento.


**Centralidade de proximidade (Closeness Centrality)**


```python
print("=== Centralidade de proximidade (Closeness) ===\n")
print("Calculando... (pode levar alguns segundos)\n")

# Calcular para uma amostra de nós (grafo grande)
closeness_cent = nx.closeness_centrality(G, wf_improved=True)
top5_close = sorted(closeness_cent.items(), key=lambda x: x[1], reverse=True)[:5]

print(" Top 5 nós por centralidade de proximidade:")
for node, score in top5_close:
    print(f"  - Nó {node}: {score:.6f}")

print("\n Interpretação:")
print("Nós com alta closeness estão 'bem localizados' — alcançam outros nós")
print("rapidamente. Ideal para localização de comércios ou serviços de emergência.")

```

    === Centralidade de proximidade (Closeness) ===
    
    Calculando... (pode levar alguns segundos)
    
     Top 5 nós por centralidade de proximidade:
      - Nó 1355441834: 0.019445
      - Nó 7828670046: 0.019363
      - Nó 1355442109: 0.019350
      - Nó 2252734352: 0.019337
      - Nó 1355441867: 0.019297
    
     Interpretação:
    Nós com alta closeness estão 'bem localizados' — alcançam outros nós
    rapidamente. Ideal para localização de comércios ou serviços de emergência.


### Extraímos listas auxiliares (para depuração)


```python
print("=== Listas auxiliares para inspeção rápida ===\n")

# Lista apenas com IDs dos nós
noi = [n for n, d in G.nodes(data=True)]

# Lista apenas com pares (origem, destino) das arestas
eoi = [(u, v) for u, v, d in G.edges(data=True)]

# Lista completa de nós com atributos
d = [d for d in G.nodes(data=True)]

print("📌 `noi` — primeiros 10 IDs de nós:")
print(noi[:10])

print("\n📌 `eoi` — primeiras 10 arestas (origem, destino):")
print(eoi[:10])

print("\n📌 `d` — primeiro nó com seus atributos:")
print(d[0])

```

    === Listas auxiliares para inspeção rápida ===
    
    📌 `noi` — primeiros 10 IDs de nós:
    [254452257, 254452316, 255289695, 255289717, 255289736, 255289745, 255289756, 256906109, 256906168, 256906169]
    
    📌 `eoi` — primeiras 10 arestas (origem, destino):
    [(254452257, 1348578073), (254452257, 1338106242), (254452316, 3094036085), (255289695, 4851291168), (255289695, 3175038197), (255289717, 3383066915), (255289717, 3383066940), (255289717, 4851291198), (255289736, 3383261979), (255289736, 5377740898)]
    
    📌 `d` — primeiro nó com seus atributos:
    (254452257, {'y': -21.7664571, 'x': -43.3484305, 'street_count': 4})


### Visualização geoespacial


```python
print("=== Mapa interativo dos bares de JF ===\n")

# Criar mapa centralizado no primeiro bar
mapa = folium.Map(location=[X[0, 0], X[0, 1]], zoom_start=13)

# Adicionar marcadores para todos os bares
for _, row in gdf.iterrows():
    popup_text = f"""
    <b>{row['Name']}</b><br>
    {row.get('Endereço', 'Endereço não informado')}<br>
    <i>{row.get('Petisco', '')}</i>
    """
    folium.Marker(
        [row['latitude'], row['longitude']],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color='red', icon='cutlery', prefix='fa')
    ).add_to(mapa)

# Adicionar botão de tela cheia
Fullscreen().add_to(mapa)

# Salvar e exibir (opcional: Salvar arquivo)
mapa.save("bares_jf.html")
print("✅ Mapa salvo como 'bares_jf.html'")
print("Abra este arquivo no navegador para visualizar interativamente.")

# Exibir no notebook (se estiver em ambiente Jupyter)
display(mapa)
```

### Caminhos mínimos entre bares (exemplo conceitual)


```python
print("=== Exemplo: Distância entre dois bares ===\n")

# Selecionar dois bares aleatórios
if len(gdf) >= 2:
    bar1 = gdf.iloc[0]
    bar2 = gdf.iloc[1]
    
    # Encontrar nós mais próximos de cada bar
    node1 = ox.nearest_nodes(G, bar1['longitude'], bar1['latitude'])
    node2 = ox.nearest_nodes(G, bar2['longitude'], bar2['latitude'])
    
    print(f"Bar 1: {bar1['Name']} → Nó: {node1}")
    print(f"Bar 2: {bar2['Name']} → Nó: {node2}")
    
    try:
        # Calcular caminho mais curto (tempo)
        caminho = nx.shortest_path(G, node1, node2, weight='travel_time')
        tempo = nx.shortest_path_length(G, node1, node2, weight='travel_time')
        distancia = nx.shortest_path_length(G, node1, node2, weight='length')
        
        print(f"\n🚗 Caminho mais curto:")
        print(f"  - Tempo estimado: {tempo/60:.1f} minutos")
        print(f"  - Distância: {distancia/1000:.2f} km")
        print(f"  - Número de ruas no percurso: {len(caminho)-1}")
    except nx.NetworkXNoPath:
        print("❌ Não há caminho viável entre estes dois bares na rede.")
else:
    print("Dados insuficientes para exemplo.")

```

    === Exemplo: Distância entre dois bares ===
    
    Bar 1: ADEGA BAR → Nó: 3306236011
    Bar 2: BAR DIAS → Nó: 1338603116
    
    🚗 Caminho mais curto:
      - Tempo estimado: 12.0 minutos
      - Distância: 11.34 km
      - Número de ruas no percurso: 87


**Considerações finais**

Principais descobertas:
  - Rede viária com 13,086 nós e 30,461 arestas
  - Densidade extremamente baixa (0.000178) — típica de malhas urbanas
  - Média de grau: 4.66 (mediana: 6)
  - 7,055 nós com alta conectividade (grau ≥ 6)
  - 12 bares estão localizados em interseções movimentadas
  - Centralidades calculadas (grau, betweenness, closeness)


**Referências** 

Danchev, V. (2021). Reproducible Data Science with Python<br>
NetworkX Documentation: https://networkx.org/<br>
OSMnx Documentation: https://osmnx.readthedocs.io/
