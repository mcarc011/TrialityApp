import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import time

def nodeinfo(fstr):
    if '.' in fstr:
        fstr = ''.join([t for t in fstr if not t.isalpha()])
        fstr = [t for t in fstr.split('.')]
        return [fstr[0] +'.',fstr[1]+'.']
    else:
        fstr = ','.join([t for t in fstr if not t.isalpha()])
        return [t for t in fstr.split(',')]

def readxf(dict2read):
    JEn = {}
    for key in dict2read:
        terms = '&'.join(dict2read[key])
        terms = terms.replace('+','')
        terms = terms.replace('-','')
        JEn[key] = terms.split('&')
    dict2read = JEn.copy()

    fieldnames = set()
    for fermi in dict2read:
        for term in dict2read[fermi]:
            for field in term.split(','):
                field = field.replace('+','')
                field = field.replace('-','')
                fieldnames.add(field)
        fieldnames.add(fermi)
        
    N1 = [int(nodeinfo(n)[0].replace('.','')) for n in fieldnames]
    N2 = [int(nodeinfo(n)[1].replace('.','')) for n in fieldnames]
    N = max(N1 + N2)
    Xread,Fread = np.meshgrid(np.zeros(N),np.zeros(N))

    ndictinfo = {}
    for field, relations in dict2read.items():
        ndictinfo[field] = nodeinfo(field)
        for term in relations:
            term = term.replace('+','')
            term = term.replace('-','')
            for t in term.split(','):
                ndictinfo[t] = nodeinfo(t)

    for field,val in ndictinfo.items():
        val[0] = val[0].replace('.','')
        val[1] = val[1].replace('.','')
        a,b = int(val[0])-1,int(val[1])-1
        if field in dict2read:
            Fread[a,b] += 1
            Fread[b,a] += 1
        else:
            Xread[a,b] += 1
    return Xread,Fread

    
def invertchiral(je):
    jnew = {}
    for fermi in je:
        tlist = []
        for terms in je[fermi]:
            tstr = []
            sgn = terms[0]
            terms = terms[1:]
            for field in terms.split(','):
                a,b = nodeinfo(field)[0],nodeinfo(field)[1]
                tlbl = ''.join([s for s in field if s.isalpha()])
                tstr += [tlbl+b+a]
            tstr.reverse()
            tlist += [sgn+','.join(tstr)]
        fa,fb = nodeinfo(fermi)[0],nodeinfo(fermi)[1] 
        flbl = ''.join([s for s in fermi if s.isalpha()])
        jnew[flbl+fb+fa] = tlist
    return jnew

def Triality(JEinput,p):

    def signfix(termi):
        a = termi.count('-')
        termi = termi.replace('-','')
        termi = termi.replace('+','')
        if a%2==0:
            sgn = '+'
        else:
            sgn = '-'
        return sgn + termi
    
    def generate_label(current_label):
        # Check if the current label is empty or not a string
        if not current_label or not isinstance(current_label, str):
            return "A"  # Start with 'A' if the current label is empty or not a string
        
        # Convert the current label to uppercase
        current_label = current_label.upper()
        
        # Get the last character of the current label
        last_char = current_label[-1]
        
        # If the last character is 'Z', add 'A' to the end and increment the previous character
        if last_char == 'Z':
            new_label = current_label[:-1] + 'AA'
        else:
            # Increment the last character
            new_char = chr(ord(last_char) + 1)
            new_label = current_label[:-1] + new_char
        
        return new_label 
    
    def labelgen(nkey,test):
        tlabel = 'A'
        fulllabel = tlabel+nkey
        while fulllabel in test:
            tlabel = generate_label(tlabel)
            fulllabel = tlabel+nkey
        return fulllabel
    
    def w2JandE(W):
        JEterms = {}
        for term in W:
            fermi = term.split(',')[0][2:]
            xfield = term[0] + ','.join(term.split(',')[1:])

            if '+' in xfield and '-' in xfield:
                xfield = xfield.replace('+','')
                xfield = xfield.replace('-','')
                xfield = '-'+xfield
            if xfield.count('+')==2 or xfield.count('-')==2 :
                xfield = xfield.replace('+','')
                xfield = xfield.replace('-','')
                xfield = '+'+xfield
            if fermi not in JEterms:
                JEterms[fermi] = [xfield]
            else:
                JEterms[fermi] += [xfield]

        for fermi in JEterms:
            J,E = [],[]
            for term in JEterms[fermi]:
                if nodeinfo(fermi)[1] == nodeinfo(term[1:].split(',')[0])[0]:
                    J += [term]
                else:
                    E += [term]
            # if '-' in J[0]:
            #     J = [J[1],J[0]]
            # if '-' in E[0]:
            #     E = [E[1],E[0]]
            JEterms[fermi] = J+E
        return JEterms

    Chiralfields = set()
    Fermifields = set()
    conjFermifields = set()
    nodeinfodict = {}   
    
    JEpm = JEinput.copy()
    JEn = {}
    for key in JEinput:
        terms = '&'.join(JEinput[key])
        terms = terms.replace('+','')
        terms = terms.replace('-','')
        JEn[key] = terms.split('&')
    JEinput = JEn.copy()

    for term in JEinput.items():
        for field in ','.join(term[1]).split(','):
            Chiralfields.add(field)
            nodeinfodict[field] = nodeinfo(field)
        Fermifields.add(term[0])
        ninfo = nodeinfo(term[0])
        conjFermifields.add(''.join([s for s in term[0] if s.isalpha()]) + ninfo[1]+ninfo[0])
        nodeinfodict[term[0]] = [ninfo[0],ninfo[1]]

    if '.' in list(JEinput.keys())[0]:
        RelChiralfields = {f for f in Chiralfields if str(p+1)+'.' in nodeinfodict[f]}
        RelFermifields = {f for f in Fermifields if str(p+1)+'.' in nodeinfodict[f]}
    else:
        RelChiralfields = {f for f in Chiralfields if str(p+1) in nodeinfodict[f]}
        RelFermifields = {f for f in Fermifields if str(p+1) in nodeinfodict[f]}
    incChiralFields = {f for f in RelChiralfields if nodeinfodict[f][1] == str(p+1) or nodeinfodict[f][1] == str(p+1)+'.'}
    outChiralFields = {f for f in RelChiralfields if nodeinfodict[f][0] == str(p+1) or nodeinfodict[f][0] == str(p+1)+'.'}

    fieldnames = [x for x in Chiralfields] + [f for f in Fermifields] + [f for f in conjFermifields]
    relabels = {}
    CompositeTerms = {}
    CubicTerms = []

    for xj in outChiralFields:
        relabels[xj] = '~'+xj
        fieldnames.append(''.join([s for s in xj if s.isalpha()]) + nodeinfodict[xj][1] + nodeinfodict[xj][0])
    for fi in RelFermifields:
        if nodeinfodict[fi][1] == str(p+1) or nodeinfodict[fi][1] == str(p+1)+'.' :
            relabels['~'+fi] = ''.join([s for s in fi if s.isalpha()]) + nodeinfodict[fi][0] + nodeinfodict[fi][1]
            fieldnames.remove(''.join([s for s in fi if s.isalpha()]) + nodeinfodict[fi][1] + nodeinfodict[fi][0])
        else:
            relabels['~'+fi] = ''.join([s for s in fi if s.isalpha()]) + nodeinfodict[fi][1] + nodeinfodict[fi][0]
            fieldnames.remove(''.join([s for s in fi if s.isalpha()]) + nodeinfodict[fi][0] + nodeinfodict[fi][1])

    for xi in incChiralFields:
        relabels[xi] = ''.join([s for s in xi if s.isalpha()]) + nodeinfodict[xi][1] + nodeinfodict[xi][0]
        fieldnames.append(relabels[xi])
        fieldnames.remove(xi)
        for xj in outChiralFields:
            nkeyi = nodeinfodict[xi][0] + nodeinfodict[xj][1]
            nlabel = labelgen(nkeyi,fieldnames)
            fieldnames.append(nlabel)
            CompositeTerms[xi+','+xj] = nlabel
            CubicTerms += ['+'+relabels[xj]+','+relabels[xi]+','+nlabel]
        for fi in RelFermifields:
            if nodeinfodict[fi][1] ==str(p+1) or nodeinfodict[fi][1] ==str(p+1)+'.':
                nkeyi = nodeinfodict[xi][0] + nodeinfodict[fi][0]
                conjnkeyi = nodeinfodict[fi][0] + nodeinfodict[xi][0]
            else:
                nkeyi = nodeinfodict[xi][0] + nodeinfodict[fi][1]
                conjnkeyi = nodeinfodict[fi][1] + nodeinfodict[xi][0]
            nlabel = labelgen(nkeyi,fieldnames)
            fieldnames.append(nlabel)
            fieldnames.append(''.join([s for s in nlabel if s.isalpha()]) + conjnkeyi)
            CompositeTerms[xi+',~'+fi] = '~'+nlabel
            CubicTerms += ['+~'+nlabel+','+relabels['~'+fi]+','+relabels[xi]]

    W = []
    Wc = []
    sgndic = {'-':'+','+':'-'}
    for key,relation in JEpm.items():
        for term in relation:
            W += [signfix('~'+key +','+ term)]
            if str(p+1) in nodeinfodict[key] or str(p+1)+'.' in nodeinfodict[key]:
                copyterm = True
                for xi in incChiralFields:
                    if xi in term:
                        copyterm = False
                        break
                if copyterm:
                    term = signfix('~'+key +','+ term)
                    fields = term[1:].split(',')
                    for xi in incChiralFields:
                        W += [term[0]+CompositeTerms[xi+','+fields[0]]+','
                               +CompositeTerms[xi+','+fields[1]]+','+','.join(fields[2:])]
                        Wc += [(CompositeTerms[xi+','+fields[0]][1:],
                                CompositeTerms[xi+','+fields[1]]+','+','.join(fields[2:]))]
                                          
    for wi,term in enumerate(W):
        savecycle = False
        cycle = term[1:].split(',')
        cycleterm = term[1:].split(',')

        if str(p+1) in term:
            for i in range(len(cycle)):
                cycleterm = cycle[i:] + cycle[:i]
                if ','.join(cycleterm[:2]) in CompositeTerms:
                    savecycle = True
                    break

        relabelterm = []
        ckey = ','.join(cycleterm[:2])
        cycleterm = ','.join(cycleterm)

        if savecycle:
            cycleterm = cycleterm.replace(ckey,CompositeTerms[ckey])

        for t in cycleterm.split(','):
            if t in relabels:
                relabelterm += [relabels[t]]
            else:
                relabelterm += [t]
                
        cycleterm = ','.join(relabelterm)
        
        cycleterm = cycleterm.split(',')
        cyclei = [ci for ci,c in enumerate(cycleterm) if '~' in c][0]
        W[wi] = term[0] + ','.join(cycleterm[cyclei:]+cycleterm[:cyclei])

    sgndic = {'+':'-','-':'+'}
        
    for cterm in CubicTerms:
        sgn = sgndic[cterm.split(',')[0][0]]
        cf1,cf2 = cterm.split(',')[0][2:],cterm.split(',')[1]
        icf1 = ''.join([s for s in cf1 if s.isalpha()]) + nodeinfo(cf1)[1] + nodeinfo(cf1)[0]
        for term in W:
            f1,f2 = term.split(',')[0][2:],term.split(',')[1]
            if '~'+cf1 in term and nodeinfo(f2)[0]==nodeinfo(cf2)[0]:
                sgn = term.split(',')[0][0]
                break
            if '~'+icf1 in term and nodeinfo(f2)[0]==nodeinfo(cf2)[0]:
                sgn = term.split(',')[0][0]
                break
        W += [sgndic[sgn] + cterm[1:]]

    def finddoublemass(Winput):
        massivefield = []
        doublemass = {}
        for term in Winput:
            if len(term.split(','))==2:
                f1,f2 = term.split(',')[0][1:],term.split(',')[1]
                if f1 in massivefield:
                    doublemass[f1] = []
                if f2 in massivefield:
                    doublemass[f2] = []
                massivefield += [f1,f2]

        for term in Winput:
            for field in doublemass:
                if field in term and len(term.split(','))==2:
                    pair = [f for f in term.split(',') if field not in f][0]
                    doublemass[field] += [pair[1:]]
        reldmass ={}
        for field in doublemass:
            reldmass[doublemass[field][0]] = '+'+doublemass[field][1]
            reldmass[doublemass[field][1]] = '-'+doublemass[field][0]
        return reldmass,doublemass

    def superpotential(jetest):
        Wc = []
        for key,relation in jetest.items():
            for ti, term in enumerate(relation):
                term = term.replace('+','')
                term = term.replace('-','')
                pm = '+-+-+-+-'
                Wc += [pm[ti]+'~'+key +','+ term]
        return Wc
    
    def simplifyterms(jetest):   
        for fermi in jetest:
            tnew = []
            tsgn = []
            for tival in jetest[fermi]:
                if tival[1:] not in tnew:
                    tnew += [tival[1:]]
                    tsgn += [tival[0]]
                if tival[1:] in tnew:
                    ti = tnew.index(tival[1:])
                    if tsgn[ti] != tival[0]:
                        tnew.remove(tival[1:])
                        tsgn.remove(tsgn[ti])
            jetest[fermi] = [tsgn[i]+tnew[i] for i in range(len(tnew))]
        return jetest 

    W = [signfix(wi) for wi in W]
    JEint = w2JandE(W)
    #jbefore = w2JandE(W)
    for wt in Wc:
        fermi = wt[0]
        terml = JEint[fermi]
        wcterm = wt[1].split(',')
  
        #find opposite term
        terml.remove([t for t in terml if wt[1] in t][0])
        for t in terml:
            if nodeinfo(t.split(',')[0])[0] == nodeinfo(wcterm[0])[0]:
                break
        terml += [sgndic[t[0]]+wt[1]]
        JEint[fermi] = terml

    Wn = []
    reldmassdict,dmassfield = finddoublemass(W)

    xmassfield = []
    for key in JEint:
        xmassfield +=[t[1:] for t in JEint[key] if len(t.split(','))==1]

    if len(dmassfield) !=0:
        # Xn, Fn = readxf(JEinput)
        # return np.array(Xn),np.array(Fn), JEinput
        for field, relations in w2JandE(W).items():
            if '~'+field in reldmassdict:
                for term in relations:
                    Wn += [reldmassdict['~'+field]+','+term]
            for term in relations:
                Wn += [term[0]+'~'+field+','+term[1:]]
        W = superpotential(simplifyterms(w2JandE(Wn)))
        reldmassdict,dmassfield = finddoublemass(W)

    integrateout = {}
    signdic = {'++':'-','+-':'+','-+':'+','--':'-'}
    massivefermi = set()
    for fermi,relations in w2JandE(W).items():
        J,E = [],[]
        for term in relations:
            if nodeinfo(fermi)[1] == nodeinfo(term[1:].split(',')[0])[0]:
                J += [term]
            else:
                E += [term]
        if len([j for j in J if len(j.split(','))==1])>0:
            lhs = [j for j in J if len(j.split(','))!=1]
            rhs = [j for j in J if len(j.split(','))==1][0]
            lhstxt = ''
            for l in lhs:
                lhstxt += signdic[l[0]+rhs[0]]+l[1:]
            if len(lhs)>1:
                lhstxt = '('+lhstxt+')'
            integrateout[rhs[1:]] = lhstxt
            massivefermi.add(fermi)
        if len([e for e in E if len(e.split(','))==1])>0:
            lhs = [e for e in E if len(e.split(','))!=1]
            rhs = [e for e in E if len(e.split(','))==1][0]
            lhstxt = ''
            for l in lhs:
                lhstxt += signdic[l[0]+rhs[0]]+l[1:]
            if len(lhs)>1:
                lhstxt = '('+lhstxt+')'
            integrateout[rhs[1:]] = lhstxt
            massivefermi.add(fermi)
    # if len(massivefermi)>0:
    #     print('Intout')

    for key in integrateout:
        W= ('&'.join(W).replace(key,integrateout[key])).split('&')

    Wd = []
    for term in W:
        if '(' in term:
            a,b = term.index('('),term.index(')')
            rhs,mid,lhs = term[:a],term[a:b],term[b:]
            rhs = rhs.replace('(','')
            lhs = lhs.replace(')','')
            mid = mid.replace('(','')
            mid = mid.replace(')','')
            midsgn = mid[1:]
            if '-' in midsgn:
                midsgn = '-'
            if '+' in midsgn:
                midsgn = '+'
            t1,t2 = mid[1:].split(midsgn)[0],mid[1:].split(midsgn)[1]
            term1,term2 = rhs+mid[0]+t1+lhs,rhs+midsgn+t2+lhs
            Wd += [signfix(term1),signfix(term2)]
        else:
            Wd += [signfix(term)]    

    JEnew = simplifyterms(w2JandE(Wd))
    JEint = {}
    for key in JEnew:
        if key not in massivefermi:
            terms = '&'.join(JEnew[key])
            JEint[key] = terms.split('&')
    W = superpotential(JEint)

    Xn, Fn = readxf(JEint)
    return Xn, Fn, JEint

def get_edges(input_mtx: np.array) -> dict:
    """Determine what edges are connect, based on matrix elements

    Args:
        input_mtx (np.array): Matrix describing connected elements.

    Returns:
        dict: Ordered dictionary with keys of nodes in connected order and values of "weights"/"number of arrows".
    """
    edge_labels_dict = {}
    for i in range(len(input_mtx)):
        for j in range(len(input_mtx[i])):
            if i != j:
                number_of_arrows = input_mtx[i][j]
                if number_of_arrows != 0:
                    start_node = i+1
                    end_node = j+1
                    edge_labels_dict[(start_node, end_node)] = int(number_of_arrows)
    return edge_labels_dict

def draw_edges_and_labels(G: nx.DiGraph, pos: dict, edge_labels_dict: dict, color: str, font_size=12, node_color="yellow", node_size=500, label_pos=0.1, **kwargs):
    """Draw the connected nodes.

    Args:
        G (nx.DiGraph): networkx graph.
        pos (dict): Positions of nodes.
        edge_labels_dict (dict): Ordered dictionary with keys of nodes in connected order and values of "weights"/"number of arrows".
        color (str): Edge and label's font color.
        font_size (int, optional): Font size of label. Defaults to 12.
        node_color (str, optional): Color of nodes. Defaults to "yellow".
        node_size (int, optional): Node size. Defaults to 500.
        label_pos (float, optional): Position of edge label along edge (0=head, 0.5=center, 1=tail). Defaults to 0.1.
    """
    nx.draw(
        G, pos, edge_color=color, width=1, linewidths=0.5,
        node_size=node_size, node_color=node_color, alpha=0.9,
        labels={node: node for node in G.nodes()},
        **kwargs
    )
    if (color=="red") | (color=="r"):
        nx.draw_networkx_edges(G, pos, arrowsize=1, edge_color=color)
        nx.draw_networkx_edge_labels(
            G, pos, label_pos=0.5,
            edge_labels=edge_labels_dict,
            font_color=color,
            font_size=font_size,
            **kwargs
        )
    else:
        nx.draw_networkx_edges(G, pos, arrowsize=20, edge_color=color)
        nx.draw_networkx_edge_labels(
            G, pos, label_pos=label_pos,
            edge_labels=edge_labels_dict,
            font_color=color,
            font_size=font_size,
            **kwargs
        )
    return

def add_edges_to_graph(G: nx.DiGraph, edge_labels_dict: dict, color: str):
    """Add edges to connected nodes on graph.

    Args:
        G (nx.DiGraph): networkx graph.
        edge_labels_dict (dict): Ordered dictionary with keys of nodes in connected order and values of "weights"/"number of arrows".
        color (str): Color of edges.
    """
    list_for_add_edges_from = list(edge_labels_dict.keys())
    G.add_edges_from(list_for_add_edges_from, color=color)
    return


def generate_quiver_diagram(input_mtx: np.array, black_edge_labels_dict: dict, black_color: str, red_edge_labels_dict: dict, red_color: str, figsize=(5,5), savefig=False):
    """Generate quiver diagrams.  Connected nodes are determined by input matricies. 

    Args:
        input_mtx (np.array): Matrix of nodes connected with black lines.
        black_edge_labels_dict (dict): Black edge ordered dictionary with keys of nodes in connected order and values of "weights"/"number of arrows".
        black_color (str): Color used for plot black edges and labels.
        red_edge_labels_dict (dict): Red edge ordered dictionary with keys of nodes in connected order and values of "weights"/"number of arrows".
        red_color (str): Color used for plot black edges and labels.
        figsize (tuple): Size of output graphic. Defaults to (12,8).
        savefig (str or False, optional): Path for saved figure.  When set to False, figure is shown but not saved. Defaults to False.
    """
    fig = plt.figure(figsize=figsize)

    # set_plotting_style("ggplot") # doesnt work with nx
    Gk = nx.DiGraph() # for black nodes / edges
    Gr = nx.DiGraph() # for red nodes / edges
    
    # NOTE: shows connected nodes...ALSO SHOWS UNCONNECTED NODES (if they exist)
    nodes = np.arange(1, len(input_mtx)+1).tolist()
    Gk.add_nodes_from(nodes)

    # # NOTE: ONLY SHOWS CONNECTED NODES
    # list_for_add_edges_from = list(edge_labels_dict.keys())
    # G.add_edges_from(list_for_add_edges_from)

    # add black edges
    add_edges_to_graph(Gk, black_edge_labels_dict, black_color)

    # add red edges
    add_edges_to_graph(Gr, red_edge_labels_dict, red_color)

    # choose node locations / layout
    pos = nx.circular_layout(Gk)

    # add edges and labels
    draw_edges_and_labels(Gk, pos, black_edge_labels_dict, black_color, label_pos=0.3)
    draw_edges_and_labels(Gr, pos, red_edge_labels_dict, red_color, label_pos=0.3)


    st.pyplot(fig)
    return

def generate_graph(xt,ft):
    black_line_mtx, red_line_mtx = xt,ft

    # generate dictionaries of edge data, for each matrix
    k_edge_labels_dict = get_edges(black_line_mtx)
    red_edge_labels_dict = get_edges(red_line_mtx)

    generate_quiver_diagram(black_line_mtx, k_edge_labels_dict, "k", red_edge_labels_dict, red_color="r", savefig='')
    


example = '''{'X31': ['+A12,X23', '-C12,Y23', '+P36,X64,P41', '-Q36,X64,Q41'],
  'Y31': ['-A12,Z23', '+B12,Y23', '+P36,Y64,P41', '-Q36,Y64,Q41'],
  'Z31': ['-B12,X23', '+C12,Z23', '+P36,Z64,P41', '-Q36,Z64,Q41'],
  'D26': ['+X64,P41,A12', '-Z64,P41,B12', '+P25,X56', '-X23,P36'],
  'E26': ['+Y64,P41,B12', '-X64,P41,C12', '+P25,Y56', '-Y23,P36'],
  'F26': ['+Z64,P41,C12', '-Y64,P41,A12', '+P25,Z56', '-Z23,P36'],
  'G26': ['+X64,Q41,A12', '-Z64,Q41,B12', '+X23,Q36', '-Q25,X56'],
  'H26': ['+Y64,Q41,B12', '-X64,Q41,C12', '+Y23,Q36', '-Q25,Y56'],
  'I26': ['+Z64,Q41,C12', '-Y64,Q41,A12', '+Z23,Q36', '-Q25,Z56'],
  'P45': ['+X56,X64', '-Z56,Y64', '+Q41,A12,Q25', '-P41,A12,P25'],
  'Q45': ['+Y56,Y64', '-X56,Z64', '+Q41,B12,Q25', '-P41,B12,P25'],
  'R45': ['+Z56,Z64', '-Y56,X64', '+Q41,C12,Q25', '-P41,C12,P25']}'''

st.markdown('# Triality Calculator #')
with st.expander('Instructions',expanded=False):
    st.markdown(''' This calculator takes in a superpotential and performs triality
on it. The input should be in the format the example below is in. 

To run the example:

1. Copy the dictionary by clicking input and then the clipboard button
to copy the example input

2. Paste it into the JE terms text area

3. Specify which node to dualize in the node text area. You can put a minus
sign to perform inverse Triality

4. Hit the Triality button and give the app a second and then hit it again
to show the new quiver and JE terms underneath the new output dropdown.

    ''')
col1,col2 = st.columns(2)

if 'xfi' not in st.session_state:
    JEex = eval(example)
    Xi,Fi = readxf(JEex)
    st.session_state['xfi'] = Xi,Fi,JEex

if 'xfi' in st.session_state:
    Xi,Fi,JEi = st.session_state['xfi']
    with col1:
        generate_graph(Xi,Fi)
        with st.expander('Input',expanded=False):
            st.write(JEi)
    if 'xfn' in st.session_state:
        Xn,Fn,JEn = st.session_state['xfn']
        with col2:
            generate_graph(Xn,Fn)
            with st.expander('Output',expanded=False):
                st.write(JEn)

JEterms = st.text_area('JE terms')
node = st.text_input('Node')
tbutton= st.button('Triality')

if tbutton:
    JEi = eval(JEterms)
    Xi,Fi = readxf(JEi)
    st.session_state['xfi'] = Xi,Fi,JEi
    if int(node)>0:
        Xn,Fn, JEn = Triality(JEi,int(node)-1)
        time.sleep(3)
    if int(node)<0:
        JEn = invertchiral(Triality(invertchiral(JEi), int(node)+1)[-1])
        Xn,Fn = readxf(JEn)
        time.sleep(3)
    st.session_state['xfn'] = Xn,Fn,JEn