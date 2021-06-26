import dash_core_components as dcc
import dash_html_components as html

from app import app

layout = html.Div(className = '', children = [

    html.Div(className = 'box', children = [
        html.H1('HEADWAYS REAL-TIME MONITORING AND ANOMALY DETECTION',className = 'title is-3'),
        html.Div(className = 'columns', children = [
            html.Div(className = 'column', children =  [
                html.Div(className = 'box', children = [
                    html.H1('MADRID EMT BUSES',className = 'subtitle is-3'),
                    html.Div(className = 'columns', children = [
                        html.Div(className = 'column is-narrow', children =  [
                            html.Div(className = '', children = [
                                dcc.Link('Line 1', className = 'button is-link is-light subtitle is-4', href = '/realtime/madrid/1')
                            ])
                        ]),
                        html.Div(className = 'column is-narrow', children =  [
                            html.Div(className = '', children = [
                                dcc.Link('Line 44', className = 'button is-link is-light subtitle is-4', href = '/realtime/madrid/44')
                            ])
                        ]),
                        html.Div(className = 'column is-narrow', children =  [
                            html.Div(className = '', children = [
                                dcc.Link('Line 82', className = 'button is-link is-light subtitle is-4', href = '/realtime/madrid/82')
                            ])
                        ]),
                        html.Div(className = 'column is-narrow', children =  [
                            html.Div(className = '', children = [
                                dcc.Link('Line 132', className = 'button is-link is-light subtitle is-4', href = '/realtime/madrid/132')
                            ])
                        ]),
                        html.Div(className = 'column is-narrow', children =  [
                            html.Div(className = '', children = [
                                dcc.Link('Line 133', className = 'button is-link is-light subtitle is-4', href = '/realtime/madrid/133')
                            ])
                        ])
                    ])
                ])
            ]),
            html.Div(className = 'column', children =  [
                html.Div(className = 'box', children = [
                    html.H1('LONDON BUSES',className = 'subtitle is-3'),
                    html.Div(className = 'columns', children = [
                        html.Div(className = 'column is-narrow', children =  [
                            html.Div(className = '', children = [
                                dcc.Link('Line 18', className = 'button is-link is-light subtitle is-4', href = '/realtime/london/18')
                            ])
                        ]),
                        html.Div(className = 'column is-narrow', children =  [
                            html.Div(className = '', children = [
                                dcc.Link('Line 25', className = 'button is-link is-light subtitle is-4', href = '/realtime/london/25')
                            ])
                        ])
                    ])
                ])
            ])
        ])
    ])
    
])