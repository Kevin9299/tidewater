import './core/TSLPatches.js';
import { App } from './App.js';
import { UI } from './ui/UI.js';
import { AppUI } from './ui/AppUI.js';

const ui = new UI();
const app = new App();
window.__ui = ui;

app.init( ( p, text ) => ui.setLoading( p, text ) ).then( async () => {

	app.ui = new AppUI( app, ui );
	ui.setLoading( 1, 'Ready' );
	await ui.hideLoader();
	app.start();
	ui.showStartOverlay( () => {

		app.input.requestLock();
		if ( app.audio ) app.audio.resume();

	} );

} ).catch( ( e ) => {

	console.error( e );
	ui.setLoading( null, 'Error: ' + e.message );

} );
