function replace_big_select_inputs(a,b,c){function d(a){var b=$(a),c={placeholder:"Click to select",closeOnSelect:!b.is("[MULTIPLE]"),dropdownAutoWidth:!0,containerCssClass:"select2-minwidth"};return a.select2(c)}jQuery.fn.select2&&(void 0===a&&(a=20),void 0===b&&(b=3e3),c=c||$("select"),c.each(function(){var c=$(this).not("[multiple]"),e=c.find("option").length;a>e||e>b||c.hasClass("no-autocomplete")||d(c)}))}function init_refresh_on_change(){$("select[refresh_on_change='true']").off("change").change(function(){var a=$(this),b=a.val(),c=a.attr("refresh_on_change_values");if(c){c=c.split(",");var d=a.attr("last_selected_value");if(-1===$.inArray(b,c)&&-1===$.inArray(d,c))return}$(window).trigger("refresh_on_change"),$(document).trigger("convert_to_values"),a.get(0).form.submit()}),$(":checkbox[refresh_on_change='true']").off("click").click(function(){var a=$(this),b=a.val(),c=a.attr("refresh_on_change_values");if(c){c=c.split(",");var d=a.attr("last_selected_value");if(-1===$.inArray(b,c)&&-1===$.inArray(d,c))return}$(window).trigger("refresh_on_change"),a.get(0).form.submit()}),$("a[confirm]").off("click").click(function(){return confirm($(this).attr("confirm"))})}window.jQuery=jQuery,window.$=jQuery,window._=_,window.Backbone=Backbone,window.panels=require("layout/panel"),_.extend(window,require("layout/modal")),window.async_save_text=require("utils/async-save-text");var POPUPMENU=require("ui/popupmenu");window.make_popupmenu=POPUPMENU.make_popupmenu,window.make_popup_menus=POPUPMENU.make_popup_menus,window.init_tag_click_function=require("ui/autocom_tagging");var TOURS=require("mvc/tours");window.init_refresh_on_change=init_refresh_on_change,$(document).ready(function(){init_refresh_on_change(),$.fn.tooltip&&($(".unified-panel-header [title]").tooltip({placement:"bottom"}),$("[title]").tooltip()),make_popup_menus(),replace_big_select_inputs(20,1500),$("a").click(function(){var a=$(this),b=parent.frames&&parent.frames.galaxy_main;if("galaxy_main"==a.attr("target")&&!b){var c=a.attr("href");c+=-1==c.indexOf("?")?"?":"&",c+="use_panels=True",a.attr("href",c),a.attr("target","_self")}return a});try{if(urlparms=_.object(_.compact(_.map(location.search.slice(1).split("&"),function(a){return a?a.split("="):void 0}))),urlparms.tour_id){var a=urlparms.tour_id;delete urlparms.tour_id;var b=$(location).attr("href"),c=_.map(Object.getOwnPropertyNames(urlparms),function(a){return"?"+[a,urlparms[a]].join("=")}).join("&");b=window.location.href.split("?")[0],"?"!==c&&(b+=c),history.pushState(null,null,b),TOURS.giveTour(a)}else if(et=JSON.parse(sessionStorage.getItem("activeGalaxyTour")),et&&(et=TOURS.hooked_tour_from_data(et),et&&et.steps)){var d=new Tour(_.extend({steps:et.steps},TOURS.tour_opts));d.init(),d.restart()}}catch(e){console.log("Tour loading failure, you'll need to restart the tour manually."+e)}});
//# sourceMappingURL=../maps/onload.js.map