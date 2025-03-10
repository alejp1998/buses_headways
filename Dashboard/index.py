import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import app_home, app_credits, app_realtime_madrid, app_realtime_london

#APP INDEX STRING
app.index_string = '''
<!DOCTYPE html>
<html class = "">
    <head>
        {%metas%}
        <title>HEADWAYS</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.1/css/bulma.min.css" integrity="sha512-ZRv40llEogRmoWgZwnsqke3HNzJ0kiI0+pcMgiz2bxO6Ew1DVBtWjVn0qjrXdT3+u+pSN36gLgmJiiQ3cQtyzA==" crossorigin="anonymous" />
        <script defer src="https://use.fontawesome.com/releases/v5.3.1/js/all.js"></script>
        <link href='https://fonts.googleapis.com/css?family=Megrim' rel='stylesheet'>
            <style>
                .logo1 {
                    color : white;
                    font-family: 'Megrim';
                    font-size: 30px;
                }
                .navlogo {
                    background: linear-gradient(to left, white, #840BBC);
                }
            </style>
        {%favicon%}
        {%css%}
    </head>
    <body class="body">
        <nav class="navbar navlogo" role="navigation" aria-label="main navigation">
          <div class="navbar-brand">
            <a class="navbar-item logo1" href="/home">HEADWAYS</a>
            <div class="navbar-burger burger" data-target="navMenu">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
          <div id="navMenu" class="navbar-menu">
            <div class="navbar-start">
              <div class="navbar-item has-dropdown is-hoverable">
                <a class="navbar-link">MADRID EMT</a>
                <div class="navbar-dropdown">
                <a class="navbar-item" href="/realtime/madrid/1">Line 1</a>
                <a class="navbar-item" href="/realtime/madrid/44">Line 44</a>
                <a class="navbar-item" href="/realtime/madrid/82">Line 82</a>
                <a class="navbar-item" href="/realtime/madrid/132">Line 132</a>
                <a class="navbar-item" href="/realtime/madrid/133">Line 133</a>
                </div>
            </div>
            <div class="navbar-start">
              <div class="navbar-item has-dropdown is-hoverable">
                <a class="navbar-link">LONDON</a>
                <div class="navbar-dropdown">
                <a class="navbar-item" href="/realtime/london/18">Line 18</a>
                <a class="navbar-item" href="/realtime/london/25">Line 25</a>
                </div>
            </div>
            <a class="navbar-item" href="/credits">Credits</a>
            </div>
          </div>
        </nav>
        <script type="text/javascript">
            document.addEventListener('DOMContentLoaded', () => {
            // Get all "navbar-burger" elements
            const $navbarBurgers = Array.prototype.slice.call(document.querySelectorAll('.navbar-burger'), 0);
            // Check if there are any navbar burgers
            if ($navbarBurgers.length > 0) {
                // Add a click event on each of them
                $navbarBurgers.forEach( el => {
                    el.addEventListener('click', () => {
                        // Get the target from the "data-target" attribute
                        const target = el.dataset.target;
                        const $target = document.getElementById(target);
                        // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
                        el.classList.toggle('is-active');
                        $target.classList.toggle('is-active');
                    });
                });
            }
            });
        </script>
        <section class="hero"
            {%app_entry%}
        </section>
        <footer height='0'>
                {%config%}
                {%scripts%}
                {%renderer%}
        </footer>
    </body>
</html>
'''

#APP LAYOUT DIV
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

#CHANGE PAGE DEPENDING ON PATHNAME
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    try :
        if (pathname == '/home') or (pathname == '/'):
            return app_home.layout
        elif pathname[0:16] == '/realtime/madrid':
            if pathname[17:] not in ['1','44','82','132','133'] :
                return html.H1('Line not available for real time analysis yet',className='title is-3')
            return app_realtime_madrid.layout
        elif pathname[0:16] == '/realtime/london':
            if pathname[17:] not in ['18','25'] :
                return html.H1('Line not available for real time analysis yet',className='title is-3')
            return app_realtime_london.layout
        else:
            return app_credits.layout
    except :
        return app_home.layout

#START THE SERVER
server = app.server
if __name__ == '__main__':
    app.run_server(port=8050, host='0.0.0.0', debug=False)
