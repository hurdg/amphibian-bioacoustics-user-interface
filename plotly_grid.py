import plotly.graph_objects as go

def plotly_grid(file_dict, spp_code, spp_name, ai_range, samp_key):
    fig = go.Figure()

    z_index_unhashable = list(reversed([list(range(i,i+5)) for i in range(0,60,5)]))
    z_index_hashable = [z for z_list in z_index_unhashable for z in z_list]
    z_vals_hashable = [round(file_dict[spp_code+'_pos'][str(z)],2) for z in z_index_hashable]
    z_vals_unhashable = [z_vals_hashable[i:i+5] for i in range(0, 60, 5)]


    selection_slot = list([0]*int(samp_key)) + [1]+ list((59-int(samp_key))*[0])
    selection_box_hashable = [selection_slot[z] for z in z_index_hashable]
    selection_box_unhashable = [selection_box_hashable[i:i+5] for i in range(0, 60, 5)]

    fig.add_trace(go.Heatmap(
                        z=selection_box_unhashable,
                        zmin = 0,
                        zmax = 1,
                        xgap = 0,
                        ygap=0,
                        opacity=1,
                        colorscale=['rgba(255,255,255,0)','rgba(0,0,0,1)'],
                        showscale=False, showlegend = False)
                        )

    fig.add_trace(go.Heatmap(
                        z=z_vals_unhashable,
                        zmin = 0,
                        zmax = 1,
                        xgap = 5,
                        ygap=5,
                        opacity=1,
                        colorscale=['rgba(255, 200, 200, 1)','rgba(255, 230, 230, 1)','rgba(230, 230, 230, 1)', 'rgba(230, 255, 230, 1)', 'rgba(200, 255, 200, 1)'],
                            colorbar=dict(
                                tick0=0,
                                dtick=1,
                                title=dict( text = f"A.I. Confidence of {spp_name} Presence",
                                            font = dict(size = 12)),
                                titleside="right",
                                len = 0.65, lenmode = 'fraction',
                                yanchor="top", y=0.88, x=0.9,
                                thickness = 0.08, thicknessmode = 'fraction'),
                        showscale=True, showlegend = False)
                        )
    


    x_grid = [x for x in range(0,5)]*12
    y_grid = [y for ys in [[yss]*5 for yss in reversed(range(0,12))] for y in ys]


    below_ai_key = [int(key) for key, value in file_dict[spp_code+'_pos'].items() if value  < ((ai_range[0]-0.01)/100)]
    x_grid_below_ai = [x_grid[below_key] for below_key in below_ai_key]
    y_grid_below_ai = [y_grid[below_key] for below_key in below_ai_key]


    fig.add_trace(go.Scatter(
        x=x_grid_below_ai,
        y= y_grid_below_ai,
        mode = 'markers',
                    marker=dict(
                    symbol = 'x-open',
                    color='maroon',
                    size=9,
                ),
        name = f"{spp_code} Absent - (Classified by A.I.)",
        hoverinfo='none', 
        showlegend = True,
        legendrank=5
    ))

    above_ai_key = [int(key) for key, value in file_dict[spp_code+'_pos'].items() if value > ((ai_range[1]+0.01)/100)]
    x_grid_above_ai = [x_grid[above_key] for above_key in above_ai_key]
    y_grid_above_ai = [y_grid[above_key] for above_key in above_ai_key]


    fig.add_trace(go.Scatter(
        x=x_grid_above_ai,
        y=y_grid_above_ai,
        mode = 'markers',
                    marker=dict(
                    symbol = 'circle-open',
                    color='green',
                    size=9,
                ),
        name = f"{spp_code} Present - (Classified by A.I.)",
        hoverinfo='none', 
        showlegend = True,
        legendrank=3
    ))

    between_ai_key = [int(key) for key, value in file_dict[spp_code+'_pos'].items() if (value <= ai_range[1]/100) and (value >= ai_range[0]/100) ]
    x_grid_between_ai = [x_grid[between_key] for between_key in between_ai_key]
    y_grid_between_ai = [y_grid[between_key] for between_key in between_ai_key]


    fig.add_trace(go.Scatter(
        x=x_grid_between_ai,
        y=y_grid_between_ai,
        mode = 'markers',
                    marker=dict(
                    symbol = 'asterisk-open',
                    color='blue',
                    size=11,
                ),
        name = "Unclassified",
        hoverinfo='none', 
        showlegend = True,
        legendrank=1
    ))


    user_transcribed = [key for key, value in file_dict['transcriber_'+spp_code].items() if isinstance(value, str) and (value !='EIM_AI')]
    user_transcribed_neg = [int(key) for key, value in file_dict[spp_code].items() if (value ==0) and (key in user_transcribed)]
    user_transcribed_pos = [int(key) for key, value in file_dict[spp_code].items() if (value ==1) and (key in user_transcribed)]    

    x_grid_user_neg = [x_grid[user_neg_key] for user_neg_key in user_transcribed_neg]
    y_grid_user_neg = [y_grid[user_neg_key] for user_neg_key in user_transcribed_neg]

    x_grid_user_pos = [x_grid[user_pos_key] for user_pos_key in user_transcribed_pos]
    y_grid_user_pos = [y_grid[user_pos_key] for user_pos_key in user_transcribed_pos]

    fig.add_trace(go.Scatter(
        x=x_grid_user_neg,
        y=y_grid_user_neg,
        mode = 'markers',
                    marker=dict(
                    symbol = 'x',
                    color='red',
                    size=13,
                    line=dict(
                        color='black',
                        width=1)                    
                ),
        name = f"{spp_code} Absent - (Classified by User)",
        hoverinfo='none', 
        showlegend = True,
        legendrank=4
    ))

    fig.add_trace(go.Scatter(
        x=x_grid_user_pos,
        y=y_grid_user_pos,
        mode = 'markers',
                    marker=dict(
                    symbol = 'circle',
                    color='#00FF00',
                    size=13,
                    line=dict(
                        color='black',
                        width=1)
                ),
        name = f"{spp_code} Present - (Classified by User)",
        hoverinfo='none', 
        showlegend = True,
        legendrank=2
    ))




    for i in range(0,60):
        fig.add_annotation(
            x=x_grid[i]+0.28, y=y_grid[i]-0.3, # position
            text=str(i), # text
            showarrow=False,
            font=dict(size=10, color="black")
        )
    
    fig.update_layout(
        autosize=True,       
        #width = 500,
        #height = 500,
        title = "",
        title_x=0.1,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(
            orientation="h",
            y=-0.015,
            x=0.2),
        yaxis=dict(title=""),         
        font=dict(
            #family="Courier New, monospace",
            #size=16,
            color="black",
            variant="small-caps",
        )
    )

    fig.update_xaxes(
        range=[-0.5,4.5],  # sets the range of xaxis
        constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
        showticklabels=False,
        tickvals = [x-0.5 for x in range(0,5)]
    )
    fig.update_yaxes(
        range=[-0.5,11.5],
        scaleanchor = "x",
        scaleratio = 1,
        showticklabels=False, 
        tickvals = [x-0.5 for x in range(0,13)]
    )

    return(fig)