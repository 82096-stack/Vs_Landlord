from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ddz.connection_manager import ConnectionManager
from ddz.game_room import SeatInfo
from ddz.models import Player
from ddz.rules import MODE_RULES

# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(title="DouDiZhu WebSocket Server")

manager = ConnectionManager()

# ============================================================
# HTML Frontend
# ============================================================

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>斗地主</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0d1b2a;color:#e0e1dd;min-height:100vh}
.container{max-width:900px;margin:0 auto;padding:20px}
h1{text-align:center;color:#f4a261;margin:20px 0}
.panel{background:#1b2838;border:1px solid #2a3a4a;border-radius:12px;padding:16px;margin:10px 0}
.panel h3{color:#e9c46a;margin-bottom:10px}
input,button{font-size:1rem;padding:8px 12px;border-radius:6px;border:1px solid #3a4a5a;background:#16212e;color:#e0e1dd;margin:4px}
button{background:#f4a261;color:#0d1b2a;font-weight:bold;cursor:pointer}
button:hover{background:#e76f51}
button:disabled{background:#555;cursor:not-allowed}
.row{display:flex;gap:6px;flex-wrap:wrap}
.row>*{flex:1;min-width:120px}
#log{background:#0a1118;border:1px solid #2a3a4a;border-radius:8px;padding:12px;height:280px;overflow-y:auto;font-family:monospace;font-size:.85rem;white-space:pre-wrap}
#hand{min-height:50px;background:#0a1118;border:1px solid #2a3a4a;border-radius:8px;padding:10px;font-family:monospace}
.card-btn{display:inline-block;background:#f4a261;color:#0d1b2a;padding:5px 8px;margin:3px;border-radius:5px;cursor:pointer;font-weight:bold;font-size:.85rem;border:2px solid transparent;user-select:none}
.card-btn.selected{border-color:#e76f51;background:#e76f51;color:#fff}
</style>
</head>
<body>
<div class="container">
<h1>斗地主 WebSocket</h1>
<div class="panel">
<h3>连接</h3>
<div class="row">
<input id="username" placeholder="昵称" maxlength="20">
<button onclick="createRoom()">创建房间</button>
<input id="joinRoomId" placeholder="房间号" maxlength="6">
<button onclick="joinRoom()">加入房间</button>
</div>
<button id="startBtn" onclick="startGame()" style="display:none;background:#2ecc71">开始游戏</button>
<div id="roomInfo" style="margin-top:8px;color:#e9c46a"></div>
</div>
<div class="panel" id="gamePanel" style="display:none">
<h3>游戏面板</h3>
<div id="gameState" style="color:#9aa5b1;margin-bottom:8px"></div>
<div id="handContainer">
<label>你的手牌 <span id="cardCount">0</span> 张</label>
<div id="hand"></div>
</div>
<div class="row" style="margin-top:8px">
<button id="playBtn" onclick="playCards()" disabled>出牌</button>
<button id="passBtn" onclick="send({type:'pass'})" disabled>过牌</button>
</div>
<div id="bidPanel" style="display:none;margin-top:8px">
<label>叫地主</label>
<div class="row" id="bidButtons"></div>
</div>
</div>
<div class="panel">
<h3>日志</h3>
<div id="log"></div>
</div>
</div>
<script>
let ws=null,roomId=null,mySeat=null,myCards=[],selected=new Set(),reqCounter=0;
function rid(){return 'r'+Date.now()+'_'+(++reqCounter)}
function L(m){const e=document.getElementById('log');const t=new Date().toLocaleTimeString();e.textContent+='['+t+'] '+m+'\\n';e.scrollTop=e.scrollHeight}
function connect(){const p=location.protocol==='https:'?'wss:':'ws:';ws=new WebSocket(p+'//'+location.host+'/ws');ws.onopen=()=>L('已连接');ws.onclose=()=>L('已断开');ws.onmessage=e=>{handle(JSON.parse(e.data))}}
function send(d){if(ws&&ws.readyState===WebSocket.OPEN){d.request_id=d.request_id||rid();ws.send(JSON.stringify(d))}}
function createRoom(){const u=document.getElementById('username').value.trim();if(!u){L('请输入昵称');return}send({type:'create_room',username:u,mode:'classic'})}
function joinRoom(){const u=document.getElementById('username').value.trim();const rid=document.getElementById('joinRoomId').value.trim().toUpperCase();if(!u||!rid){L('请输入昵称和房间号');return}send({type:'join_room',room_id:rid,username:u})}
function startGame(){send({type:'start_game'})}
function playCards(){if(selected.size===0)return;const cards=Array.from(selected).map(i=>myCards[i]);const idx=Array.from(selected).sort((a,b)=>a-b);send({type:'play_card',cards:idx,action:'play'});
selected.clear();document.querySelectorAll('.card-btn.selected').forEach(b=>b.classList.remove('selected'))}
function handle(msg){
    const type=msg.type;
    const p=msg.payload||{};
    const rid=msg.request_id||'';
    L('收到: '+type+(rid?' ['+rid+']':''));
    switch(type){
    case'room_created':roomId=p.room_id||msg.room_id;document.getElementById('roomInfo').innerHTML='房间 <b>'+roomId+'</b> 已创建';document.getElementById('gamePanel').style.display='block';break;
    case'room_joined':roomId=p.room_id||msg.room_id;mySeat=p.seat!=null?p.seat:msg.seat;document.getElementById('roomInfo').innerHTML='已加入 <b>'+roomId+'</b> 座位'+(mySeat+1);document.getElementById('gamePanel').style.display='block';break;
    case'room_state':updateRoom(p);break;
    case'game_starting':L('游戏开始! '+((p.mode_label)||''));document.getElementById('gameState').textContent='游戏中';break;
    case'game_started':L('游戏开始!');document.getElementById('gameState').textContent='游戏中';break;
    case'your_hand':case'your_cards':myCards=p.cards||[];renderHand();document.getElementById('cardCount').textContent=myCards.length;break;
    case'cards_dealt':L('标记牌: '+p.marked_card+' 底牌: '+p.bottom_count+'张');break;
    case'ask_bid':showBid(p.allowed_bids||[0,1,2,3]);break;
    case'ask_call':showCall();break;
    case'ask_rob':showRob(p.highest_bid||0);break;
    case'bid_result':L('座位'+p.seat+' '+p.player_name+': '+(p.bid===0?'不叫':p.bid+'分'));break;
    case'call_result':L('座位'+p.seat+' '+p.player_name+': '+(p.call?'叫地主':'不叫'));break;
    case'rob_result':L('座位'+p.seat+' '+p.player_name+': '+(p.rob?'抢地主':'不抢'));break;
    case'bidding_turn':break;
    case'no_bidder':L(p.message||'无人叫地主');break;
    case'redeal':L(p.message||'重新发牌');break;
    case'landlord_assigned':L('座位'+p.seat+' '+p.player_name+' 成为地主，底牌: '+(p.bottom_cards||[]).map(c=>c.label).join(' '));renderHand();break;
    case'play_turn':L('轮到 座位'+p.seat+' '+p.player_name+(p.is_opening?' (新回合)':''));break;
    case'ask_play':L(p.is_opening?'请出牌（新回合）':'请出牌，压过 '+((p.last_combo&&p.last_combo.description)||'?'));document.getElementById('playBtn').disabled=false;document.getElementById('passBtn').disabled=p.is_opening;renderHand();break;
    case'play_action':
        if(p.action==='play'){
            L('座位'+p.seat+' '+p.player_name+' 出牌 '+p.combo_display+' 剩'+p.remaining_count+'张');
        }else{
            L('座位'+p.seat+' '+p.player_name+' 过牌');
        }
        document.getElementById('playBtn').disabled=true;document.getElementById('passBtn').disabled=true;
        break;
    case'player_empty':L('座位'+p.seat+' '+p.player_name+' 手牌出完!');break;
    case'player_disconnected':L('座位'+p.seat+' '+p.username+' 断开连接');break;
    case'new_round':L('新回合: '+p.leader_name);break;
    case'reveal_result':L('座位'+p.seat+' '+p.player_name+(p.reveal?' 摊打':' 不摊打'));break;
    case'report_result':L('座位'+p.seat+' '+p.player_name+' '+p.report_label);break;
    case'game_over':L('游戏结束! 胜者: '+p.winner_name+' ('+p.winner_role+') 地主: '+p.landlord_name);document.getElementById('playBtn').disabled=true;document.getElementById('passBtn').disabled=true;break;
    case'state_snapshot':L('收到状态快照 (重连恢复)');if(p.players)updateRoomFromSnapshot(p);break;
    case'error':L('错误: '+(p.message||msg.message||''));break;
    default:L('未知消息: '+type);
    }
}
function updateRoom(p){
    const ps=(p.players||[]).map(function(x){return '座位'+x.seat+': '+(x.username||'空')+' '+(x.role==='landlord'?'[地主]':'')+' '+x.hand_size+'张'}).join(' | ');
    L('房间 '+(p.room_id||roomId)+' '+(p.state||'')+'\\n'+ps);
    if(p.state==='waiting'&&roomId){
        var filled=(p.players||[]).filter(function(x){return x.username}).length>=3;
        document.getElementById('startBtn').style.display=filled?'inline-block':'none';
    }
}
function updateRoomFromSnapshot(p){
    L('房间恢复: '+(p.room_id||roomId)+' '+p.state);
    if(p.landlord_index!=null)L('地主: 座位'+p.landlord_index);
    if(p.current_turn!=null)L('当前回合: 座位'+p.current_turn);
}
function renderHand(){var c=document.getElementById('hand');selected.clear();c.innerHTML=myCards.map(function(s,i){return '<span class="card-btn" onclick="toggle('+i+')" id="c-'+i+'">'+(s.label||s)+'</span>'}).join('')}
function toggle(i){var b=document.getElementById('c-'+i);if(selected.has(i)){selected.delete(i);b.classList.remove('selected')}else{selected.add(i);b.classList.add('selected')}}
function showBid(allowed){var p=document.getElementById('bidPanel');var btns=document.getElementById('bidButtons');p.style.display='block';btns.innerHTML=allowed.map(function(b){return '<button onclick="send({type:\\'bid\\',bid:'+b+'});document.getElementById(\\'bidPanel\\').style.display=\\'none\\'">'+(b===0?'不叫':b+'分')+'</button>'}).join('')}
function showCall(){var p=document.getElementById('bidPanel');var btns=document.getElementById('bidButtons');p.style.display='block';btns.innerHTML='<button onclick="send({type:\\'call\\',call:true});document.getElementById(\\'bidPanel\\').style.display=\\'none\\'">叫地主</button><button onclick="send({type:\\'call\\',call:false});document.getElementById(\\'bidPanel\\').style.display=\\'none\\'">不叫</button>'}
function showRob(bid){var p=document.getElementById('bidPanel');var btns=document.getElementById('bidButtons');p.style.display='block';btns.innerHTML='<button onclick="send({type:\\'rob\\',rob:true});document.getElementById(\\'bidPanel\\').style.display=\\'none\\'">抢地主</button><button onclick="send({type:\\'rob\\',rob:false});document.getElementById(\\'bidPanel\\').style.display=\\'none\\'">不抢</button>'}
connect();
</script>
</body>
</html>"""


# ============================================================
# HTTP Endpoints
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    return INDEX_HTML


@app.get("/health")
async def health():
    return {"status": "ok", "rooms": len(manager.rooms)}


# ============================================================
# WebSocket Endpoint
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    current_room_id: str | None = None
    current_seat: int | None = None
    current_username: str = ""

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({
                    "type": "error",
                    "room_id": current_room_id or "",
                    "timestamp": "",
                    "request_id": None,
                    "payload": {"message": "Invalid JSON"},
                })
                continue

            msg_type = data.get("type", data.get("action", ""))
            request_id = data.get("request_id")

            # ---- room management ----

            if msg_type == "create_room":
                username = (data.get("username") or "player").strip()[:30]
                mode = data.get("mode", "classic")
                if mode not in MODE_RULES:
                    mode = "classic"
                room = manager.create_room(mode, username)
                seat = room.add_player(username)
                if seat is None:
                    await ws.send_json({
                        "type": "error",
                        "room_id": "",
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "Failed to join room"},
                    })
                    continue
                current_room_id = room.room_id
                current_seat = seat
                current_username = username
                room.seats[seat].ws = ws
                room._start_sender(seat)
                manager.register_ws(ws, room.room_id, seat, username)

                await room._send_to(seat, "room_created", {
                    "room_id": room.room_id,
                    "seat": seat,
                }, request_id=request_id)
                await room._broadcast("room_state", room.public_room_state()["payload"])

            elif msg_type == "join_room":
                room_id = data.get("room_id", "").strip().upper()
                username = (data.get("username") or "player").strip()[:30]
                room = manager.get_room(room_id)
                if room is None:
                    await ws.send_json({
                        "type": "error",
                        "room_id": room_id,
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "房间不存在"},
                    })
                    continue
                if room.state != "waiting":
                    await ws.send_json({
                        "type": "error",
                        "room_id": room_id,
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "游戏已开始"},
                    })
                    continue
                seat = room.add_player(username)
                if seat is None:
                    await ws.send_json({
                        "type": "error",
                        "room_id": room_id,
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "房间已满"},
                    })
                    continue
                current_room_id = room_id
                current_seat = seat
                current_username = username
                room.seats[seat].ws = ws
                room._start_sender(seat)
                manager.register_ws(ws, room_id, seat, username)

                await room._send_to(seat, "room_joined", {
                    "room_id": room_id,
                    "seat": seat,
                }, request_id=request_id)
                await room._broadcast("room_state", room.public_room_state()["payload"])

            elif msg_type == "start_game":
                room = manager.get_room(current_room_id) if current_room_id else None
                if room is None:
                    await ws.send_json({
                        "type": "error",
                        "room_id": current_room_id or "",
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "未在房间中"},
                    })
                    continue
                if room.host_username != (room.seats.get(current_seat, SeatInfo(username="")).username if current_seat is not None else ""):
                    await ws.send_json({
                        "type": "error",
                        "room_id": current_room_id,
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "只有房主可以开始"},
                    })
                    continue
                if not room.all_seats_filled():
                    room.fill_with_ai()
                if not any(s.is_human and s.connected for s in room.seats.values()):
                    await ws.send_json({
                        "type": "error",
                        "room_id": current_room_id,
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "至少需要一名真人玩家"},
                    })
                    continue
                await room.start_game()

            elif msg_type == "leave_room":
                if current_room_id:
                    room = manager.get_room(current_room_id)
                    if room and current_seat is not None:
                        await room._stop_sender(current_seat)
                        room.remove_player(room.seats[current_seat].username)
                        manager.unregister_ws(ws)
                        await room._broadcast("room_state", room.public_room_state()["payload"])
                        human_count = sum(1 for s in room.seats.values() if not s.is_human or s.connected)
                        if human_count == 0:
                            manager.remove_room(current_room_id)
                current_room_id = None
                current_seat = None
                current_username = ""

            # ---- game actions ----

            elif msg_type in ("bid", "play_card", "pass", "call", "rob", "reveal", "report"):
                room = manager.get_room(current_room_id) if current_room_id else None
                if room is None or current_seat is None:
                    await ws.send_json({
                        "type": "error",
                        "room_id": current_room_id or "",
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "未在游戏中"},
                    })
                    continue

                if msg_type == "bid":
                    room.handle_response(current_seat, {
                        "bid": data.get("bid", data.get("score", 0)),
                    })
                elif msg_type == "call":
                    room.handle_response(current_seat, {
                        "call": data.get("call", False),
                    })
                elif msg_type == "rob":
                    room.handle_response(current_seat, {
                        "rob": data.get("rob", False),
                    })
                elif msg_type == "reveal":
                    room.handle_response(current_seat, {
                        "reveal": data.get("reveal", False),
                    })
                elif msg_type == "report":
                    room.handle_response(current_seat, {
                        "report": data.get("report", False),
                    })
                elif msg_type == "pass":
                    room.handle_response(current_seat, {
                        "action": "pass",
                        "cards": [],
                    })
                elif msg_type == "play_card":
                    room.handle_response(current_seat, {
                        "action": data.get("action", "play"),
                        "cards": data.get("cards", []),
                    })

            # ---- recovery / reconnection ----

            elif msg_type == "reconnect":
                token = data.get("recovery_token", "")
                result = await manager.reconnect_player(token, ws)
                if result is None:
                    await ws.send_json({
                        "type": "error",
                        "room_id": "",
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "重连失败，token无效或已过期。"},
                    })
                    continue
                current_room_id = result["room_id"]
                # Reconstruct seat from snapshot
                for p in result["payload"].get("players", []):
                    if p.get("username") == data.get("username"):
                        current_seat = p["seat"]
                        current_username = p["username"]
                        break
                await ws.send_json(result)

            elif msg_type == "request_snapshot":
                room = manager.get_room(current_room_id) if current_room_id else None
                if room is None:
                    await ws.send_json({
                        "type": "error",
                        "room_id": current_room_id or "",
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "未在房间中"},
                    })
                    continue
                snapshot = room.full_state_snapshot(for_seat=current_seat)
                await ws.send_json(snapshot)

            else:
                await ws.send_json({
                    "type": "error",
                    "room_id": current_room_id or "",
                    "timestamp": "",
                    "request_id": request_id,
                    "payload": {"message": f"未知操作: {msg_type}"},
                })

    except WebSocketDisconnect:
        pass
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup on disconnect via ConnectionManager (generates recovery token)
        if current_room_id is not None and current_seat is not None:
            try:
                await manager.handle_disconnect(ws)
            except Exception:
                pass


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
